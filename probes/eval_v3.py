"""Path 2 v2 unified eval harness (Phase 0 contract).

Three responsibilities:
  - Translate a (config, benchmark) pair into per-row scoring + pathology
    detection (delegated to probes.extractors).
  - Append-and-fsync each row to a JSONL file so partials survive
    pod crashes and are rsync-friendly.
  - Aggregate JSONL into a per-config summary JSON with the four
    headline metrics: accuracy_smart_v2 (or accuracy for ARC/BBH),
    truncation_rate, loop_rate, parse_rate.

This module is intentionally pure-Python, no torch / transformers
imports at module load. The model + hook layer lives in the entry
point (experiments/path2_v2_eval.py); this module just consumes
completion strings.

Per-row JSONL schema (GSM8K cell):
  {
    "idx": int, "question": str, "gold": any, "gold_int": int|None,
    "completion": str, "n_gen_tokens": int,
    "pred_smart_v2": int|None, "pred_legacy": int|None,
    "correct_smart_v2": bool, "correct_legacy": bool,
    "truncated": bool, "loop_flag": bool,
    "t_gen_seconds": float
  }

Per-config summary fields (GSM8K):
  accuracy_smart_v2, accuracy_legacy, extractor_lift,
  truncation_rate, loop_rate, parse_rate (smart_v2 returned non-None),
  n_problems, mean_gen_tokens, total_gen_seconds, mean_t_gen_seconds,
  pathology_flag (truncation_rate>0.5 or loop_rate>0.5).

``truncation_rate`` semantics (Phase 1, post-disambiguation):
  ``n_tok >= max_new_tokens`` is the raw row-level fact recorded as
  ``truncated``. The summary metric is the principled subset:
  ``truncated AND pred_smart_v2 is None`` --- i.e., the model hit the
  cap AND the harness couldn't extract any answer. A row that hits
  the cap but yields a parseable (correct or wrong) number is an
  *accuracy* observation, not truncation; the original definition
  conflated those, which made gates fire on verbose-but-correct
  generations (idx=314: model emits "Answer: 30" matching gold, then
  continues into self-checking and trips the cap). Phase 1 N=200 1A
  saw 14/200 = 7 % under the old definition vs 0/200 = 0 % under the
  new one; all 14 had extractable numbers.
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable

from . import extractors


# ---------------------------------------------------------------------------
# Config registry
# ---------------------------------------------------------------------------
#
# Phase 1 cells (sanity gates) live here; Phase 2/3/4 will append.
# Each config dict carries everything needed to schedule the cell:
#   name           --- unique key (used for JSONL filename)
#   prompt         --- "C2" or "8shot-CoT"
#   block          --- (start, end) inclusive, or None for no hook
#   r              --- loop count (1 = identity at the hook level)
#   ple_strategy   --- "every-iter" or "iter1-only"; only consulted
#                       when block is not None
#   notes          --- free-form provenance
#
# The same name in different phases must mean the same thing.
# Phase 1 and Phase 3 both have `baseline-C2`; both use the same
# config. Phase 4's `W5-r4` is a new entry, not a mutation of W5-r8.

CONFIGS: dict[str, dict] = {
    # Sanity-gate baselines (Phase 1, also re-used as anchors in Phase 3).
    "baseline-C2": {
        "prompt": "C2",
        "block": None,
        "r": 1,
        "ple_strategy": "every-iter",
        "notes": "Path 1 anchor 71.6% on N=500 GSM8K; gate band [0.65, 0.82].",
    },
    "baseline-8shot-control": {
        "prompt": "8shot-CoT",
        "block": None,
        "r": 1,
        "ple_strategy": "every-iter",
        "notes": (
            "Path 1 plan 5 anchor: smart_v2 ~ 30 %. Single-turn + "
            "'#### N' exemplar marker. NOT the round-5 bridge --- "
            "round 5 used multi-turn + 'The answer is N.' (see "
            "baseline-8shot-round5)."
        ),
    },
    "baseline-8shot-round5": {
        "prompt": "8shot-CoT-r5",
        "block": None,
        "r": 1,
        "ple_strategy": "every-iter",
        "stop_strings": ["\nQ:", "\nQuestion:"],
        "notes": (
            "Round 5 anchor: legacy ~ 54.8 %. Multi-turn alternating "
            "exemplars + 'The answer is N.' marker + stop_strings. "
            "Mirrors probes.mode_round5._format_gsm8k_prompt_chat."
        ),
    },
    "W5-r1": {
        "prompt": "C2",
        "block": (15, 19),
        "r": 1,
        "ple_strategy": "every-iter",
        "notes": "r=1 must be a token-for-token no-op vs baseline-C2.",
    },
    "W5-r8": {
        "prompt": "C2",
        "block": (15, 19),
        "r": 8,
        "ple_strategy": "every-iter",
        "notes": "Round 5's primary block. Phase 1 smoke target: loop_rate<0.95, exits 0.",
    },
}


# Phase 2: single-layer reasoning probes. One config per decoder layer
# at r=8 with the C2 prompt; structural pre-flight uses L17-r1 instead.
# Generated programmatically so we don't paste 35 near-identical dicts.
for _L in range(35):
    CONFIGS[f"L{_L:02d}-r8"] = {
        "prompt": "C2",
        "block": (_L, _L),                # start == end -> single layer
        "r": 8,
        "ple_strategy": "every-iter",
        "notes": f"phase 2: single-layer r=8 probe at decoder layer {_L}",
    }
del _L

CONFIGS["L17-r1"] = {                     # phase 2 pre-flight only
    "prompt": "C2",
    "block": (17, 17),
    "r": 1,
    "ple_strategy": "every-iter",
    "notes": "phase 2 pre-flight: single-layer r=1 must token-match baseline-C2.",
}


def get_config(name: str) -> dict:
    if name not in CONFIGS:
        raise ValueError(
            f"unknown config {name!r}; registered: {sorted(CONFIGS)}"
        )
    return dict(CONFIGS[name])  # copy to prevent mutation


# ---------------------------------------------------------------------------
# JSONL append-and-fsync
# ---------------------------------------------------------------------------

def append_row(path: Path, row: dict) -> None:
    """Atomically append one row to a JSONL file and fsync.

    Round 4/5's pattern: never lose more than the row currently in
    flight. The rsync watcher pulls partial JSONL fine; this is the
    rounding step that makes that work.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False))
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def existing_idxs(path: Path) -> set:
    """Return the set of ``idx`` values already in the JSONL.

    Resume contract: rerunning a cell that already finished re-reads
    the JSONL and skips problems whose idx is present.
    """
    return {row["idx"] for row in read_jsonl(path)}


# ---------------------------------------------------------------------------
# Per-config summarisation
# ---------------------------------------------------------------------------

def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def summarise_gsm8k(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    n = len(rows)
    if n == 0:
        return {"n_problems": 0}
    n_smart = sum(1 for r in rows if r["correct_smart_v2"])
    n_legacy = sum(1 for r in rows if r["correct_legacy"])
    # See module docstring: truncation = "harness couldn't surface an
    # answer", not "model emitted 512 tokens". Cap-hit rows that still
    # produce a parseable number belong in the accuracy bucket.
    n_trunc = sum(
        1 for r in rows
        if r["truncated"] and r["pred_smart_v2"] is None
    )
    n_loop = sum(1 for r in rows if r["loop_flag"])
    n_parsed = sum(1 for r in rows if r["pred_smart_v2"] is not None)
    mean_tokens = sum(r["n_gen_tokens"] for r in rows) / n
    total_t = sum(r.get("t_gen_seconds", 0.0) for r in rows)
    return {
        "n_problems": n,
        "accuracy_smart_v2": n_smart / n,
        "accuracy_smart_v2_ci95": list(_wilson_ci(n_smart, n)),
        "accuracy_legacy": n_legacy / n,
        "accuracy_legacy_ci95": list(_wilson_ci(n_legacy, n)),
        "extractor_lift": (n_smart - n_legacy) / n,
        "truncation_rate": n_trunc / n,
        "loop_rate": n_loop / n,
        "parse_rate": n_parsed / n,
        "mean_gen_tokens": mean_tokens,
        "total_gen_seconds": total_t,
        "mean_t_gen_seconds": total_t / n if n else 0.0,
        "pathology_flag": (n_trunc / n > 0.5) or (n_loop / n > 0.5),
    }


def summarise_arc(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    n = len(rows)
    if n == 0:
        return {"n_problems": 0}
    n_correct = sum(1 for r in rows if r["correct"])
    n_trunc = sum(1 for r in rows if r["truncated"])
    n_loop = sum(1 for r in rows if r["loop_flag"])
    n_parsed = sum(1 for r in rows if r["pred_letter"] is not None)
    mean_tokens = sum(r["n_gen_tokens"] for r in rows) / n
    total_t = sum(r.get("t_gen_seconds", 0.0) for r in rows)
    return {
        "n_problems": n,
        "accuracy": n_correct / n,
        "accuracy_ci95": list(_wilson_ci(n_correct, n)),
        "truncation_rate": n_trunc / n,
        "loop_rate": n_loop / n,
        "parse_rate": n_parsed / n,
        "mean_gen_tokens": mean_tokens,
        "total_gen_seconds": total_t,
        "mean_t_gen_seconds": total_t / n if n else 0.0,
        "pathology_flag": (n_trunc / n > 0.5) or (n_loop / n > 0.5),
    }


def summarise_bbh(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    n = len(rows)
    if n == 0:
        return {"n_problems": 0}
    n_correct = sum(1 for r in rows if r["correct"])
    n_trunc = sum(1 for r in rows if r["truncated"])
    n_loop = sum(1 for r in rows if r["loop_flag"])
    mean_tokens = sum(r["n_gen_tokens"] for r in rows) / n
    total_t = sum(r.get("t_gen_seconds", 0.0) for r in rows)
    by_task = {}
    for r in rows:
        t = r.get("task", "?")
        by_task.setdefault(t, [0, 0])
        by_task[t][1] += 1
        if r["correct"]:
            by_task[t][0] += 1
    return {
        "n_problems": n,
        "accuracy": n_correct / n,
        "accuracy_ci95": list(_wilson_ci(n_correct, n)),
        "truncation_rate": n_trunc / n,
        "loop_rate": n_loop / n,
        "mean_gen_tokens": mean_tokens,
        "total_gen_seconds": total_t,
        "mean_t_gen_seconds": total_t / n if n else 0.0,
        "pathology_flag": (n_trunc / n > 0.5) or (n_loop / n > 0.5),
        "per_task_accuracy": {t: c / nt for t, (c, nt) in by_task.items()},
    }


SUMMARISERS = {
    "gsm8k": summarise_gsm8k,
    "arc-c": summarise_arc,
    "bbh-lite": summarise_bbh,
}


def summarise(rows: Iterable[dict], benchmark: str) -> dict:
    if benchmark not in SUMMARISERS:
        raise ValueError(
            f"unknown benchmark {benchmark!r}; "
            f"registered: {sorted(SUMMARISERS)}"
        )
    return SUMMARISERS[benchmark](rows)


# ---------------------------------------------------------------------------
# Result file paths
# ---------------------------------------------------------------------------

def cell_jsonl_path(*, output_dir: Path, benchmark: str, config_name: str) -> Path:
    """Each cell writes to ``<output_dir>/<benchmark>__<config>.jsonl``."""
    return Path(output_dir) / f"{benchmark}__{config_name}.jsonl"


def cell_summary_path(*, output_dir: Path, benchmark: str, config_name: str) -> Path:
    return Path(output_dir) / f"{benchmark}__{config_name}.summary.json"


# ---------------------------------------------------------------------------
# Pretty-print summary table for a directory
# ---------------------------------------------------------------------------

def print_summary_table(output_dir: Path, *, benchmark: str | None = None) -> None:
    """Aggregate every JSONL in ``output_dir``, print a compact table.

    Used by ``--summarize-only``. Designed to be safe to run *while*
    a job is still writing rows: re-reads JSONL each call.
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        print(f"(no results dir at {output_dir})")
        return

    rows_by_cell = {}
    for p in sorted(output_dir.glob("*.jsonl")):
        if not p.name.endswith(".jsonl"):
            continue
        try:
            bench, cfg = p.stem.split("__", 1)
        except ValueError:
            continue
        if benchmark and bench != benchmark:
            continue
        rows = read_jsonl(p)
        rows_by_cell[(bench, cfg)] = rows

    if not rows_by_cell:
        print(f"(no JSONL cells in {output_dir})")
        return

    print(f"=== Path 2 v2 summary ({output_dir}) ===")
    header = f"{'bench':>9}  {'config':>22}  {'n':>4}  {'acc':>7}  {'trunc':>6}  {'loop':>6}  {'parse':>6}  flag"
    print(header)
    for (bench, cfg), rows in sorted(rows_by_cell.items()):
        s = summarise(rows, bench)
        if s["n_problems"] == 0:
            print(f"{bench:>9}  {cfg:>22}  {0:>4}  {'-':>7}  {'-':>6}  {'-':>6}  {'-':>6}  -")
            continue
        if bench == "gsm8k":
            acc = s["accuracy_smart_v2"]
            parse = s["parse_rate"]
        else:
            acc = s["accuracy"]
            parse = s.get("parse_rate", float("nan"))
        flag = "PATH" if s["pathology_flag"] else ""
        print(
            f"{bench:>9}  {cfg:>22}  {s['n_problems']:>4}  "
            f"{acc:>6.1%}  {s['truncation_rate']:>5.1%}  "
            f"{s['loop_rate']:>5.1%}  "
            f"{parse:>5.1%}  {flag}"
        )
