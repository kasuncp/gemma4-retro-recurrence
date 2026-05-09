"""Path 2 v2 Phase 2 runner --- per-layer reasoning map.

Sequentially runs:
  P0_L17_r1_token_match  pre-flight (N=20)
    + N=50 GSM8K cells L00..L34 at r=8 with the C2 prompt on the IT
    model. Each cell is its own subprocess so a single-layer crash
    doesn't kill the rest.

After all 36 cells finish, applies probes.phase2_gates.classify_layer
to bucket each layer (preserved/degraded/broken), decides Phase 3's
launch mode, and writes ``phase2_summary.json`` + the heat-map PNG +
``phase2_layer_map.csv``.

Modes:
  (default)             --- pre-flight, then 35 layer cells, then aggregate.
  --summarize-only      --- read existing JSONL/summary JSON, recompute
                            the map. CPU only; safe with no torch.
  --skip-cell L<NN>     --- skip individual layer cells (repeatable).
                            Useful when one layer crashes and you want
                            to fix that layer without re-running 34.
  --screen-mode         --- optional fast-path: run all 35 at N=20 first,
                            then re-run the top 10 at N=50. Saves ~40 %
                            wall but doubles plan complexity. Default off.
  --shard <i>/<n>       --- for parallel pods: cell L<k> runs only when
                            ``k % n == i``. Three pods at --shard 0/3 1/3
                            2/3 cuts wall to ~2.5 h.
  --with-ppl-bridge     --- forwarded to path2_v2_eval.py. Default off.
                            Adds ~5 min/cell.
  --dry-run             --- print what would run; do not execute.

Idempotency: each subprocess respects --no-resume independently. By
default it appends to existing JSONL (skips already-done idxs). Pass
--no-resume on the runner to wipe + re-run every cell.

The aggregator runs whether or not we did any subprocess work, so
running with --summarize-only on a CPU laptop is the standard way to
re-render the heat map after a sync-down from the pod.
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
from probes import phase2_gates as gates
from probes.eval_v3 import (
    cell_jsonl_path, cell_summary_path, read_jsonl, summarise,
)
from probes.introspect import get_layer_attention_info  # noqa: F401  (used via runtime model)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "results" / "path_2_depth_recurrence_v2" / "phase2"
PHASE1_OUTPUT = REPO_ROOT / "results" / "path_2_depth_recurrence_v2" / "phase1"
ROUND2C_JSON = REPO_ROOT / "results" / "path_2_depth_recurrence" / "results_round2c_full_map.json"
IT_MODEL = "google/gemma-4-E2B-it"
NUM_LAYERS = 35


# ---------------------------------------------------------------------------
# Cell schedule
# ---------------------------------------------------------------------------

def _layer_cells(*, n: int = 50) -> list[dict]:
    """Return the 35 single-layer L<NN>-r8 cells."""
    out = []
    for L in range(NUM_LAYERS):
        out.append({
            "name": f"L{L:02d}",
            "layer": L,
            "kind": "eval",
            "config": f"L{L:02d}-r8",
            "benchmark": "gsm8k",
            "n": n,
            "model_id": IT_MODEL,
        })
    return out


PRE_FLIGHT = {
    "name": "P0_L17_r1_token_match",
    "kind": "eval",
    "config": "L17-r1",
    "benchmark": "gsm8k",
    "n": 20,
    "model_id": IT_MODEL,
}


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--phase1-output-dir", type=Path, default=PHASE1_OUTPUT,
        help="Where the cached baseline-C2 JSONL from Phase 1 lives. "
             "The pre-flight token-match compares against it.",
    )
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--skip-cell", action="append", default=[],
                   help="Cell name(s) to skip; can be repeated. e.g. L17.")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--with-ppl-bridge", action="store_true")
    p.add_argument(
        "--screen-mode", action="store_true",
        help="Fast-path: N=20 across all 35, then top 10 re-run at N=50.",
    )
    p.add_argument(
        "--shard", default=None,
        help="Run shard <i>/<n>: cell L<k> runs only when k %% n == i.",
    )
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-plots", action="store_true",
        help="Skip PNG generation. CSV is still written. Use when "
             "matplotlib isn't installed.",
    )
    p.add_argument(
        "--top-k-screen", type=int, default=10,
        help="With --screen-mode, how many top layers from the N=20 pass "
             "to re-run at N=50.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Subprocess builders
# ---------------------------------------------------------------------------

def _eval_subprocess_argv(*, cell, args, n_override: Optional[int] = None):
    n = cell["n"] if n_override is None else n_override
    argv = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "path2_v2_eval.py"),
        "--phase", "layer-map",
        "--config", cell["config"],
        "--benchmark", cell["benchmark"],
        "--n", str(n),
        "--seed", str(args.seed),
        "--max-new-tokens", str(args.max_new_tokens),
        "--model-id", cell["model_id"],
        "--dtype", args.dtype,
        "--output-dir", str(args.output_dir),
    ]
    if args.no_resume:
        argv.append("--no-resume")
    if args.with_ppl_bridge:
        argv.append("--with-ppl-bridge")
    return argv


def _token_match_argv(*, args):
    """Pre-flight: compare phase2's L17-r1 JSONL against phase1's
    baseline-C2 JSONL.
    """
    left = cell_jsonl_path(
        output_dir=args.phase1_output_dir, benchmark="gsm8k",
        config_name="baseline-C2",
    )
    right = cell_jsonl_path(
        output_dir=args.output_dir, benchmark="gsm8k",
        config_name="L17-r1",
    )
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
# Sharding + skip
# ---------------------------------------------------------------------------

def _parse_shard(spec: Optional[str]) -> Optional[tuple[int, int]]:
    if not spec:
        return None
    try:
        i_str, n_str = spec.split("/")
        i, n = int(i_str), int(n_str)
    except Exception as e:  # pragma: no cover - argparse will catch most
        raise SystemExit(f"--shard must be of form i/n; got {spec!r} ({e})")
    if n <= 0 or not (0 <= i < n):
        raise SystemExit(f"--shard {i}/{n} invalid (need 0 <= i < n, n > 0)")
    return i, n


def _select_layer_cells(cells: list[dict], *, shard, skip: set[str]) -> list[dict]:
    out = []
    for c in cells:
        if c["name"] in skip:
            continue
        if shard is not None:
            i, n = shard
            if c["layer"] % n != i:
                continue
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Per-cell summary loaders
# ---------------------------------------------------------------------------

def _summary_for_eval_cell(*, output_dir: Path, cell: dict) -> dict:
    jsonl = cell_jsonl_path(
        output_dir=output_dir, benchmark=cell["benchmark"],
        config_name=cell["config"],
    )
    rows = read_jsonl(jsonl)
    return summarise(rows, cell["benchmark"])


def _read_summary_json(*, output_dir: Path, cell: dict) -> Optional[dict]:
    """Return the per-cell summary.json (which may carry base_ppl_bridge)."""
    p = cell_summary_path(
        output_dir=output_dir, benchmark=cell["benchmark"],
        config_name=cell["config"],
    )
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _token_match_info(*, args) -> dict:
    """Run token_match.compare in-process. Returns ``compare`` info dict."""
    from experiments.path2_v2_token_match import compare
    left = cell_jsonl_path(
        output_dir=args.phase1_output_dir, benchmark="gsm8k",
        config_name="baseline-C2",
    )
    right = cell_jsonl_path(
        output_dir=args.output_dir, benchmark="gsm8k",
        config_name="L17-r1",
    )
    if not left.is_file() or not right.is_file():
        return {
            "reason": "missing JSONL",
            "left_exists": left.is_file(), "right_exists": right.is_file(),
            "left_path": str(left), "right_path": str(right),
        }
    code, info = compare(left, right, limit=20)
    info["exit_code"] = code
    return info


# ---------------------------------------------------------------------------
# Round 2c lookup (informational; mirrors the eval-side helper)
# ---------------------------------------------------------------------------

_ROUND2C_CACHE: dict | None = None


def _load_round2c() -> dict:
    global _ROUND2C_CACHE
    if _ROUND2C_CACHE is not None:
        return _ROUND2C_CACHE
    if not ROUND2C_JSON.is_file():
        _ROUND2C_CACHE = {}
        return _ROUND2C_CACHE
    try:
        _ROUND2C_CACHE = json.loads(ROUND2C_JSON.read_text())
    except Exception:
        _ROUND2C_CACHE = {}
    return _ROUND2C_CACHE


def _round2c_ppl_for(layer: int, *, r: int = 8, ple_mode: str = "vanilla"):
    d = _load_round2c()
    for cell in d.get("cells", []):
        if (
            cell.get("layer") == layer
            and cell.get("r") == r
            and cell.get("ple_mode") == ple_mode
        ):
            return cell.get("ppl")
    return None


def _round2c_layer_metadata(layer: int) -> dict:
    """Pull attention_type / is_kv_consumer for a layer from round 2c."""
    d = _load_round2c()
    for entry in d.get("layer_metadata", []):
        if entry.get("layer") == layer:
            return {
                "attention_type": entry.get("attention_type"),
                "is_kv_consumer": entry.get("is_kv_consumer"),
            }
    return {"attention_type": None, "is_kv_consumer": None}


def _depth_tertile(layer: int, num_layers: int = NUM_LAYERS) -> str:
    third = num_layers / 3.0
    if layer < third:
        return "early"
    if layer < 2 * third:
        return "mid"
    return "late"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(args) -> dict:
    """Produce the phase 2 summary dict from on-disk JSONL + summary JSON."""
    cells = _layer_cells()

    # Pre-flight.
    pre_info = _token_match_info(args=args)
    pre_passed = (
        pre_info.get("matches", -1) == pre_info.get("shared", 0)
        and pre_info.get("shared", 0) > 0
    )

    cells_report: dict = {}
    buckets = {"preserved": [], "degraded": [], "broken": []}

    for c in cells:
        s = _summary_for_eval_cell(output_dir=args.output_dir, cell=c)
        bucket, reason = gates.classify_layer(s)
        meta = _round2c_layer_metadata(c["layer"])
        meta["depth_tertile"] = _depth_tertile(c["layer"])

        cell_record = {
            "summary": s,
            "metadata": meta,
            "bucket": bucket,
            "bucket_reason": reason,
        }

        # Round 2c base ppl bridge (informational; the per-cell
        # base_ppl_record carries the runtime delta when --with-ppl-bridge
        # was used).
        r2c_ppl = _round2c_ppl_for(c["layer"], r=8, ple_mode="vanilla")
        if r2c_ppl is not None:
            cell_record["round2c_base_ppl"] = float(r2c_ppl)

        sj = _read_summary_json(output_dir=args.output_dir, cell=c)
        if sj is not None and "base_ppl_bridge" in sj:
            br = sj["base_ppl_bridge"]
            cell_record["base_ppl"] = br.get("base_ppl")
            if "delta_ppl" in br:
                cell_record["delta_ppl"] = br["delta_ppl"]
            if "warn" in br:
                cell_record["base_ppl_warn"] = br["warn"]

        cells_report[c["name"]] = cell_record
        buckets[bucket].append(c["name"])

    decision = gates.decide_phase3_launch(buckets)

    report = {
        "phase": "phase2-layer-reasoning-map",
        "model_ids": {"it": IT_MODEL},
        "preflight": {
            "L17_r1_token_match": {
                "shared": pre_info.get("shared", 0),
                "matches": pre_info.get("matches"),
                "mismatches": pre_info.get("mismatches"),
                "passed": pre_passed,
                "info": pre_info,
            },
        },
        "cells": cells_report,
        "buckets": buckets,
        "phase3_launch": decision,
    }
    return report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(report: dict) -> None:
    cells = report["cells"]
    buckets = report["buckets"]
    decision = report["phase3_launch"]

    print()
    print("=== Phase 2 layer reasoning map "
          "(N=50/layer, r=8, C2 prompt, IT model) ===")
    header = (
        f"{'layer':<6}{'acc':>6}  {'loop':>6}  {'trunc':>6}  "
        f"{'mean_tok':>8}  {'attn':<8}  {'kv_cons':<7}  bucket"
    )
    print(header)
    for name, c in sorted(cells.items()):
        s = c["summary"]
        if s.get("n_problems", 0) == 0:
            print(f"{name:<6}  {'-':>5}  {'-':>5}  {'-':>5}  "
                  f"{'-':>7}  {'-':<8}  {'-':<7}  {c['bucket']}")
            continue
        meta = c.get("metadata", {})
        attn = (meta.get("attention_type") or "?")[:8]
        kv = str(meta.get("is_kv_consumer"))[:5]
        print(
            f"{name:<6}{s['accuracy_smart_v2']:>6.2f}  "
            f"{s['loop_rate']:>6.2f}  {s['truncation_rate']:>6.2f}  "
            f"{s['mean_gen_tokens']:>8.0f}  {attn:<8}  {kv:<7}  "
            f"{c['bucket']}"
        )

    print()
    print("=== Bucket counts ===")
    for k in ("preserved", "degraded", "broken"):
        items = buckets.get(k, [])
        print(f"{k:<10}: {len(items):>2}  ({items})")

    print()
    print("=== Phase 3 launch decision ===")
    print(f"mode: {decision['mode']}")
    print(f"anchors: {decision['anchors']}")
    print(f"rationale: {decision['rationale']}")

    pre = report["preflight"]["L17_r1_token_match"]
    print()
    print(f"pre-flight L17-r1 token match: "
          f"matches={pre.get('matches')}/{pre.get('shared')}  "
          f"passed={pre.get('passed')}")


# ---------------------------------------------------------------------------
# Output: JSON, CSV, plot
# ---------------------------------------------------------------------------

def _emit_summary_json(args, report: dict) -> Path:
    out = Path(args.output_dir) / "phase2_summary.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {out}")
    return out


def _emit_csv_and_plot(args, report: dict) -> None:
    """Lazy import of the plotter so --no-plots / no-matplotlib paths work."""
    try:
        from experiments.path2_v2_layer_map_plot import (
            write_csv, render_png,
        )
    except Exception as e:
        print(f"WARN: could not import layer_map_plot module: {e}")
        return

    csv_path = Path(args.output_dir) / "phase2_layer_map.csv"
    write_csv(report=report, path=csv_path)
    print(f"Wrote {csv_path}")

    if args.no_plots:
        print("--no-plots set; skipping PNG.")
        return
    png_path = Path(args.output_dir) / "phase2_layer_map.png"
    try:
        render_png(report=report, path=png_path)
        print(f"Wrote {png_path}")
    except Exception as e:
        print(f"WARN: PNG render failed ({e}); CSV is still on disk.")


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------

def _run_preflight(args) -> int:
    """Run L17-r1 cell, then token-match against phase1's baseline-C2."""
    cell = PRE_FLIGHT
    rc = _run(
        _eval_subprocess_argv(cell=cell, args=args),
        dry_run=args.dry_run, label=cell["name"] + " (eval)",
    )
    if rc != 0:
        print(f"WARN: pre-flight eval cell exited {rc}; continuing to "
              "token-match for diagnostics.")
    rc_tm = _run(
        _token_match_argv(args=args),
        dry_run=args.dry_run, label=cell["name"] + " (token-match)",
    )
    if rc_tm != 0:
        print(
            "HALT: pre-flight token-match failed. install_block_loop_hooks "
            "is mishandling start==end. Fix the hook before running the "
            "35-layer sweep."
        )
    return rc_tm


def _run_layer_cells(args, *, n_override: Optional[int] = None,
                     subset: Optional[list[dict]] = None) -> int:
    cells = subset if subset is not None else _layer_cells()
    skip = set(args.skip_cell)
    shard = _parse_shard(args.shard)
    todo = _select_layer_cells(cells, shard=shard, skip=skip)
    if shard is not None:
        i, n = shard
        print(f"\n--shard {i}/{n}: running {len(todo)} of {len(cells)} cells")
    if skip:
        print(f"--skip-cell: {sorted(skip)}")

    for cell in todo:
        rc = _run(
            _eval_subprocess_argv(cell=cell, args=args, n_override=n_override),
            dry_run=args.dry_run, label=cell["name"],
        )
        if rc != 0:
            print(f"WARN: {cell['name']} returned {rc}; "
                  "continuing (Phase 2 finishes all cells).")
    return 0


def _screen_pick_top_k(args, *, k: int) -> list[dict]:
    """After an N=20 first pass, return the top-k layer cells by
    accuracy_smart_v2. Uses on-disk JSONL.
    """
    cells = _layer_cells(n=20)
    scored = []
    for c in cells:
        s = _summary_for_eval_cell(output_dir=args.output_dir, cell=c)
        if s.get("n_problems", 0) == 0:
            continue
        scored.append((s["accuracy_smart_v2"], c))
    scored.sort(key=lambda kv: kv[0], reverse=True)
    top = [c for _acc, c in scored[:k]]
    print(f"\nscreen-mode: picked top {len(top)} of {len(scored)} layers "
          f"by accuracy_smart_v2: {[c['name'] for c in top]}")
    return top


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
    report = aggregate(args)
    report["wall_seconds_total"] = 0
    _emit_summary_json(args, report)
    _emit_csv_and_plot(args, report)
    print_table(report)
    decision = report["phase3_launch"]
    pre = report["preflight"]["L17_r1_token_match"]
    if not pre.get("passed"):
        return 1
    if decision["mode"] == "halt":
        return 1
    return 0


def _run_all(args) -> int:
    t_start = time.time()

    rc = _run_preflight(args)
    if rc != 0 and not args.dry_run:
        # Halt only if the operator explicitly asks (we still aggregate
        # so rerunning aggregation doesn't require redoing the eval).
        # The plan says fix the hook before running 35 cells; we honour
        # that by exiting non-zero before launching the sweep.
        report = aggregate(args)
        report["wall_seconds_total"] = time.time() - t_start
        _emit_summary_json(args, report)
        _emit_csv_and_plot(args, report)
        print_table(report)
        return 1

    if args.screen_mode:
        # First pass: N=20 across all cells.
        print("\n=== screen-mode pass 1: N=20 across all 35 layers ===")
        _run_layer_cells(args, n_override=20)
        # Pick top-k, re-run at N=50.
        if not args.dry_run:
            top = _screen_pick_top_k(args, k=args.top_k_screen)
            print(f"\n=== screen-mode pass 2: N=50 on top {len(top)} layers ===")
            _run_layer_cells(args, n_override=50, subset=top)
    else:
        _run_layer_cells(args)

    if args.dry_run:
        print("\n(dry-run; aggregate not performed)")
        return 0

    report = aggregate(args)
    report["wall_seconds_total"] = time.time() - t_start
    _emit_summary_json(args, report)
    _emit_csv_and_plot(args, report)
    print_table(report)

    decision = report["phase3_launch"]
    pre = report["preflight"]["L17_r1_token_match"]
    if not pre.get("passed"):
        return 1
    if decision["mode"] == "halt":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
