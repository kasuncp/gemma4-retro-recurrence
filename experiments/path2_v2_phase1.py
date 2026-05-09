"""Path 2 v2 Phase 1 runner --- hook sanity gates.

Sequentially runs six cells, each as its own subprocess so a crash
in one cell doesn't blow away results from earlier ones:

  1A     baseline-C2          N=50 GSM8K  IT model
  1B-p1  baseline-8shot-control  N=50 GSM8K  IT model (Path 1 plan 5)
  1B-r5  baseline-8shot-round5   N=50 GSM8K  IT model (round 5 bridge)
  1C     W5-r1 + token-match vs 1A (N=20)
  1D     W5-r8                N=10 GSM8K  IT model
  1E     base perplexity smoke (Wikitext-2 r=1 vs unmodified)

After all cells, applies the gates from probes.phase1_gates and
emits ``phase1_summary.json`` + a console table. Exit non-zero on
any gate failure.

Modes:
  (default)         --- run every cell; budget ~45 min on a 4090.
  --summarize-only  --- skip subprocesses; just re-evaluate gates
                        from existing JSONL/JSON in --output-dir.
                        CPU only, runs in <1 s; safe with no torch.
  --skip-cell NAME  --- don't (re-)run NAME; useful after fixing one
                        cell and re-running the rest.
  --dry-run         --- print what would be run, don't execute.

Idempotency: each subprocess respects --no-resume independently. By
default it APPENDS to existing JSONL (skips already-done idxs). Pass
--no-resume on the runner to wipe + re-run every cell.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Importing probes triggers HF_HOME redirect; safe on CPU.
import probes  # noqa: F401
from probes import phase1_gates as gates
from probes.eval_v3 import (
    cell_jsonl_path, cell_summary_path, read_jsonl, summarise,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "results" / "path_2_depth_recurrence_v2" / "phase1"
IT_MODEL = "google/gemma-4-E2B-it"
BASE_MODEL = "google/gemma-4-E2B"

# Cells in execution order. Each cell is a dict; the runner translates
# it to a subprocess argv.
CELLS = [
    # --- 1A: N=200 disambiguates the N=50 truncation rate (was 3/50 =
    # 6 %, just over the < 5 % threshold; Wilson 95 % CI [0.013, 0.165]
    # was wide enough at N=50 to be small-sample noise). At N=200 the
    # CI tightens to about ±2.5 pp, so a real 6 % rate would clearly
    # fail and a noise-around-1 % rate would clearly pass.
    {
        "name": "1A_baseline_C2",
        "kind": "eval",
        "config": "baseline-C2",
        "benchmark": "gsm8k",
        "n": 200,
        "model_id": IT_MODEL,
    },
    # --- 1B-path1: bridges Path 1 plan 5's smart_v2 = 30 % anchor ---
    {
        "name": "1B_baseline_8shot_path1",
        "kind": "eval",
        "config": "baseline-8shot-control",
        "benchmark": "gsm8k",
        "n": 50,
        "model_id": IT_MODEL,
    },
    # --- 1B-round5: bridges round 5's legacy = 54.8 % anchor ---
    {
        "name": "1B_baseline_8shot_round5",
        "kind": "eval",
        "config": "baseline-8shot-round5",
        "benchmark": "gsm8k",
        "n": 50,
        "model_id": IT_MODEL,
    },
    # --- 1C: generate W5-r1 first, then token-match vs 1A ---
    {
        "name": "1C_W5_r1",            # produces the JSONL the gate compares
        "kind": "eval",
        "config": "W5-r1",
        "benchmark": "gsm8k",
        "n": 20,
        "model_id": IT_MODEL,
    },
    # --- 1D: W5-r8 process smoke ---
    {
        "name": "1D_W5_r8_smoke",
        "kind": "eval",
        "config": "W5-r8",
        "benchmark": "gsm8k",
        "n": 10,
        "model_id": IT_MODEL,
    },
    # --- 1E: base perplexity smoke ---
    {
        "name": "1E_base_ppl_smoke",
        "kind": "ppl",
        "model_id": BASE_MODEL,
    },
]


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--summarize-only", action="store_true",
                   help="Skip subprocesses; re-evaluate gates from on-disk JSONL.")
    p.add_argument("--skip-cell", action="append", default=[],
                   help="Cell name(s) to skip; can be repeated.")
    p.add_argument("--no-resume", action="store_true",
                   help="Wipe each cell's JSONL before its subprocess runs.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be run; do not execute.")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Subprocess builders
# ---------------------------------------------------------------------------

def _eval_subprocess_argv(*, cell, args):
    return [
        sys.executable,
        str(REPO_ROOT / "experiments" / "path2_v2_eval.py"),
        "--phase", "sanity",
        "--config", cell["config"],
        "--benchmark", cell["benchmark"],
        "--n", str(cell["n"]),
        "--seed", str(args.seed),
        "--max-new-tokens", str(args.max_new_tokens),
        "--model-id", cell["model_id"],
        "--dtype", args.dtype,
        "--output-dir", str(args.output_dir),
    ] + (["--no-resume"] if args.no_resume else [])


def _ppl_smoke_argv(*, args):
    """Cell 1E shells out to the existing round-1 mode.

    --r-values 1 only --- we just want unmodified ppl + r=1 ppl on
    layer 17 to compute drift.
    """
    out_json = args.output_dir / "1E_wikitext_smoke.json"
    return [
        sys.executable,
        str(REPO_ROOT / "experiments" / "ple_sanity_check.py"),
        "--mode", "original",
        "--target-layer", "17",
        "--r-values", "1",
        "--num-sequences", "50",
        "--max-length", "512",
        "--dtype", args.dtype,
        "--model-id", BASE_MODEL,
        "--output-json", str(out_json),
    ]


def _token_match_argv(*, args, left, right):
    return [
        sys.executable,
        str(REPO_ROOT / "experiments" / "path2_v2_token_match.py"),
        "--left", str(left),
        "--right", str(right),
        "--limit", "20",
    ]


def _run(argv, *, dry_run: bool, label: str) -> int:
    print(f"\n>>> {label}")
    print(f"    {' '.join(argv)}")
    if dry_run:
        print("    (dry-run; not executed)")
        return 0
    t0 = time.time()
    rc = subprocess.run(argv, cwd=str(REPO_ROOT)).returncode
    dt = time.time() - t0
    print(f"<<< {label} exited {rc} in {dt:.1f}s")
    return rc


# ---------------------------------------------------------------------------
# Per-cell summary loaders + gate dispatch
# ---------------------------------------------------------------------------

def _summary_for_eval_cell(*, output_dir: Path, cell: dict) -> dict:
    jsonl = cell_jsonl_path(
        output_dir=output_dir, benchmark=cell["benchmark"],
        config_name=cell["config"],
    )
    rows = read_jsonl(jsonl)
    return summarise(rows, cell["benchmark"])


def _ppl_smoke_results(output_dir: Path) -> tuple[Optional[float], Optional[float]]:
    """Read 1E_wikitext_smoke.json and return (unmodified_ppl, r1_ppl).

    Schema source: probes.mode_round1.run_original_mode emits
    ``{"unmodified": {"ppl": ...}, "results": {"<r>": {"ppl": ...}},
    "summary": [{"r": ..., "ppl": ..., "ratio": ...}], ...}``. Phase 1
    only invokes ``--r-values 1`` so ``results["1"]`` is the single
    looped pass we care about; ``summary`` is a redundant tolerated
    fallback so test fixtures that omit ``results`` still work.
    """
    p = output_dir / "1E_wikitext_smoke.json"
    if not p.is_file():
        return None, None
    d = json.loads(p.read_text())
    unmod = d.get("unmodified", {}).get("ppl")

    results = d.get("results")
    if isinstance(results, dict):
        r1_ppl = results.get("1", {}).get("ppl")
        if r1_ppl is not None:
            return unmod, r1_ppl

    for entry in d.get("summary", []):
        if entry.get("r") == 1:
            return unmod, entry.get("ppl")

    return unmod, None


def _token_match_info(*, output_dir: Path) -> dict:
    """Run token_match.compare in-process so we can record the dict."""
    from experiments.path2_v2_token_match import compare
    left = cell_jsonl_path(
        output_dir=output_dir, benchmark="gsm8k",
        config_name="baseline-C2",
    )
    right = cell_jsonl_path(
        output_dir=output_dir, benchmark="gsm8k", config_name="W5-r1",
    )
    if not left.is_file() or not right.is_file():
        return {"reason": "missing JSONL", "left_exists": left.is_file(),
                "right_exists": right.is_file()}
    code, info = compare(left, right, limit=20)
    info["exit_code"] = code
    return info


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------

def evaluate_all_gates(args, *, summaries: dict, ppl_pair: tuple,
                       token_match: dict) -> dict:
    cells_report = {}

    s_1A = summaries.get("1A_baseline_C2", {})
    p, m = gates.gate_1A(s_1A) if s_1A.get("n_problems") else (False, "no rows")
    cells_report["1A_baseline_C2"] = {
        "summary": s_1A, "gate_passed": p, "gate_message": m,
    }

    s_1Bp = summaries.get("1B_baseline_8shot_path1", {})
    p, m = gates.gate_1B_path1(s_1Bp) if s_1Bp.get("n_problems") else (False, "no rows")
    cells_report["1B_baseline_8shot_path1"] = {
        "summary": s_1Bp, "gate_passed": p, "gate_message": m,
    }

    s_1Br = summaries.get("1B_baseline_8shot_round5", {})
    p, m = gates.gate_1B_round5(s_1Br) if s_1Br.get("n_problems") else (False, "no rows")
    cells_report["1B_baseline_8shot_round5"] = {
        "summary": s_1Br, "gate_passed": p, "gate_message": m,
    }

    p, m = gates.gate_1C(token_match)
    cells_report["1C_token_match"] = {
        "info": token_match, "gate_passed": p, "gate_message": m,
    }

    s_1D = summaries.get("1D_W5_r8_smoke", {})
    p, m = gates.gate_1D(s_1D, expected_n=10) if s_1D.get("n_problems") else (False, "no rows")
    cells_report["1D_W5_r8_smoke"] = {
        "summary": s_1D, "gate_passed": p, "gate_message": m,
    }

    unmod, r1 = ppl_pair
    p, m = gates.gate_1E(unmod, r1)
    cells_report["1E_base_ppl_smoke"] = {
        "unmodified_ppl": unmod, "r1_ppl": r1,
        "gate_passed": p, "gate_message": m,
    }

    report = {
        "phase": "phase1-hook-sanity",
        "model_ids": {"it": IT_MODEL, "base": BASE_MODEL},
        "cells": cells_report,
        "all_gates_passed": gates.all_passed({"cells": cells_report}),
    }
    return report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _row(cell: str, metric: str, value: str, band: str, status: str) -> str:
    return f"{cell:<22}  {metric:<22}  {value:<10}  {band:<22}  {status}"


def print_table(report: dict) -> None:
    print()
    print("=== Phase 1 hook sanity gates ===")
    print(_row("cell", "metric", "value", "band", "status"))
    print("-" * 86)

    cells = report["cells"]

    # 1A
    s = cells["1A_baseline_C2"]["summary"]
    if s.get("n_problems", 0) > 0:
        print(_row("1A baseline-C2", "accuracy_smart_v2",
                   f"{s['accuracy_smart_v2']:.3f}",
                   f"{gates.CELL_1A_ACC_BAND}", _ok(cells["1A_baseline_C2"])))
        print(_row("", "loop_rate", f"{s['loop_rate']:.3f}",
                   f"< {gates.CELL_1A_LOOP_MAX}", ""))
        print(_row("", "truncation_rate", f"{s['truncation_rate']:.3f}",
                   f"< {gates.CELL_1A_TRUNC_MAX}", ""))
    else:
        print(_row("1A baseline-C2", "n_problems", "0", "= 50",
                   _ok(cells["1A_baseline_C2"])))

    # 1B-path1 (Path 1 plan 5 anchor on smart_v2)
    s = cells["1B_baseline_8shot_path1"]["summary"]
    if s.get("n_problems", 0) > 0:
        print(_row("1B-p1 8shot-path1", "accuracy_smart_v2",
                   f"{s['accuracy_smart_v2']:.3f}",
                   f"{gates.CELL_1B_SMART_BAND}",
                   _ok(cells["1B_baseline_8shot_path1"])))
        print(_row("", "accuracy_legacy", f"{s['accuracy_legacy']:.3f}",
                   "(informational)", ""))
        print(_row("", "loop_rate", f"{s['loop_rate']:.3f}",
                   f"< {gates.CELL_1B_PATH1_LOOP_MAX}", ""))
        print(_row("", "truncation_rate", f"{s['truncation_rate']:.3f}",
                   f"< {gates.CELL_1B_PATH1_TRUNC_MAX}", ""))
    else:
        print(_row("1B-p1 8shot-path1", "n_problems", "0", "= 50",
                   _ok(cells["1B_baseline_8shot_path1"])))

    # 1B-round5 (round 5 anchor on legacy)
    s = cells["1B_baseline_8shot_round5"]["summary"]
    if s.get("n_problems", 0) > 0:
        print(_row("1B-r5 8shot-round5", "accuracy_legacy",
                   f"{s['accuracy_legacy']:.3f}",
                   f"{gates.CELL_1B_LEGACY_BAND}",
                   _ok(cells["1B_baseline_8shot_round5"])))
        print(_row("", "accuracy_smart_v2",
                   f"{s['accuracy_smart_v2']:.3f}",
                   "(informational)", ""))
        print(_row("", "loop_rate", f"{s['loop_rate']:.3f}",
                   f"< {gates.CELL_1B_ROUND5_LOOP_MAX}", ""))
        print(_row("", "truncation_rate", f"{s['truncation_rate']:.3f}",
                   f"< {gates.CELL_1B_ROUND5_TRUNC_MAX}", ""))
    else:
        print(_row("1B-r5 8shot-round5", "n_problems", "0", "= 50",
                   _ok(cells["1B_baseline_8shot_round5"])))

    # 1C
    info = cells["1C_token_match"]["info"]
    shared = info.get("shared", 0)
    if "matches" in info:
        matches = info["matches"]
    elif "mismatches" in info and shared:
        matches = shared - len(info["mismatches"])
    else:
        matches = "?"
    print(_row("1C token-match", "matches", f"{matches}/{shared}", "= 20/20",
               _ok(cells["1C_token_match"])))

    # 1D
    s = cells["1D_W5_r8_smoke"]["summary"]
    if s.get("n_problems", 0) > 0:
        print(_row("1D W5-r8 smoke", "n_problems",
                   f"{s['n_problems']}", "= 10",
                   _ok(cells["1D_W5_r8_smoke"])))
        print(_row("", "loop_rate", f"{s['loop_rate']:.3f}",
                   f"< {gates.CELL_1D_LOOP_MAX}", ""))
        print(_row("", "mean_gen_tokens",
                   f"{s['mean_gen_tokens']:.0f}", "> 0", ""))
    else:
        print(_row("1D W5-r8 smoke", "n_problems", "0", "= 10",
                   _ok(cells["1D_W5_r8_smoke"])))

    # 1E
    c = cells["1E_base_ppl_smoke"]
    unmod = c.get("unmodified_ppl")
    r1 = c.get("r1_ppl")
    if unmod is not None and r1 is not None:
        drift = abs(r1 - unmod) / unmod
        print(_row("1E base ppl drift", "rel_drift", f"{drift:.2e}",
                   f"< {gates.CELL_1E_DRIFT_MAX:.0e}", _ok(c)))
    else:
        print(_row("1E base ppl drift", "missing JSON", "-", "-", _ok(c)))

    print()
    if report["all_gates_passed"]:
        print("ALL GATES PASSED. Proceed to Phase 2.")
    else:
        print("HALT: one or more gates failed:")
        for name, c in report["cells"].items():
            if not c.get("gate_passed"):
                print(f"  {name}: {c.get('gate_message', '')}")


def _ok(cell_dict: dict) -> str:
    return "PASS" if cell_dict.get("gate_passed") else "FAIL"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize_only:
        return _summarize_only(args)

    return _run_all(args)


def _summarize_only(args) -> int:
    summaries = {}
    for cell in CELLS:
        if cell["kind"] != "eval":
            continue
        summaries[cell["name"]] = _summary_for_eval_cell(
            output_dir=args.output_dir, cell=cell,
        )
    ppl_pair = _ppl_smoke_results(args.output_dir)
    token_match = _token_match_info(output_dir=args.output_dir)
    report = evaluate_all_gates(
        args, summaries=summaries, ppl_pair=ppl_pair,
        token_match=token_match,
    )
    _emit_report(args, report)
    return 0 if report["all_gates_passed"] else 1


def _run_all(args) -> int:
    skip = set(args.skip_cell)
    t_start = time.time()
    for cell in CELLS:
        if cell["name"] in skip:
            print(f"\n>>> SKIP {cell['name']}")
            continue
        if cell["kind"] == "eval":
            argv = _eval_subprocess_argv(cell=cell, args=args)
            rc = _run(argv, dry_run=args.dry_run, label=cell["name"])
        elif cell["kind"] == "ppl":
            argv = _ppl_smoke_argv(args=args)
            rc = _run(argv, dry_run=args.dry_run, label=cell["name"])
        else:
            print(f"unknown cell kind: {cell['kind']}")
            return 2
        if rc != 0:
            print(f"WARN: {cell['name']} subprocess returned {rc}; "
                  f"continuing to gate evaluation so partial results are "
                  f"still summarised.")

    if args.dry_run:
        print("\n(dry-run; gates not evaluated)")
        return 0

    summaries = {}
    for cell in CELLS:
        if cell["kind"] != "eval":
            continue
        summaries[cell["name"]] = _summary_for_eval_cell(
            output_dir=args.output_dir, cell=cell,
        )
    ppl_pair = _ppl_smoke_results(args.output_dir)
    token_match = _token_match_info(output_dir=args.output_dir)
    report = evaluate_all_gates(
        args, summaries=summaries, ppl_pair=ppl_pair,
        token_match=token_match,
    )
    report["wall_seconds_total"] = time.time() - t_start
    _emit_report(args, report)
    return 0 if report["all_gates_passed"] else 1


def _emit_report(args, report: dict) -> None:
    out = args.output_dir / "phase1_summary.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {out}")
    print_table(report)


if __name__ == "__main__":
    sys.exit(main())
