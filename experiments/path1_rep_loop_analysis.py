"""Path 1 --- plan 3: repetition-loop isolation (analysis-only).

Pure CPU re-analysis of existing plan-1 and plan-2 JSONLs. Classifies every
completion (and every individual axis-B sampled chain) into one of four
mutually-exclusive outcome buckets:

    correct, terminated_wrong, repetition_loop, truncated_no_answer

then computes per-cell adjusted accuracies under two counterfactuals
(lenient and A3-imputed upper bound), tests length-dependence and
sampling-dependence of the rep-loop rate, breaks down axis-B voted-wrong
problems into vote-on-rep categories, and lists the top sticky problem
indices --- the GSM8K problems that trigger rep-loops across the most cells.

The repetition regex is pinned to the plan-3 canonical version
``r'(.{10,60})\\1{2,}'`` and is NOT changed mid-analysis. A 30-completion
random sample of flagged rep-loops is dumped to ``flagged_sample_30.json``
for manual inspection of the false-positive rate; a heuristic
auto-classification (number of repetitions of the captured substring) is
also reported as a directional signal --- the manual sample remains the
authoritative number for the writeup.

Default --- runs over the JSONLs already on disk:
    python path1_rep_loop_analysis.py

Outputs land under ``results/path_1_cot_tokens/plan3/``:
    - results_plan3.json          (per-cell counts, adjusted accuracies, etc.)
    - path1_rep_loops_by_cell.csv (single CSV for the figure generator)
    - results_plan3.png           (multi-panel plot, skipped if matplotlib missing)
    - flagged_sample_30.json      (30 random rep-loop completions for manual FP review)

Pip install for the figure step:
    pip install matplotlib

Decision-rule output: prints the pre-registered finding letter A/B/C/D/E
based on adjusted-accuracy gaps and the axis-B chain-level rep-loop rate
relative to A3.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ---- canonical inputs --------------------------------------------------------

PLAN1_CELLS = "results/path_1_cot_tokens/plan1_preview_n100/cells"
PLAN2_CELLS = "results/path_1_cot_tokens/plan2/cells"
RESULTS_SUBDIR = "results/path_1_cot_tokens/plan3"

# (cell_name, jsonl_path, kind, source_plan, axis, max_new_tokens, k)
# kind: 'completion' = single completion per problem (plan1 + axis A)
#       'voted'      = list of k chains + voted_pred (axis B)
CELLS = [
    ("plan1.it_cot_n100",
     f"{PLAN1_CELLS}/it__cot__0000_0100.jsonl",
     "completion", "plan1", "it_cot", 512, 1),
    ("plan1.base_cot_n100",
     f"{PLAN1_CELLS}/base__cot__0000_0100.jsonl",
     "completion", "plan1", "base_cot", 512, 1),
    ("plan2.A1_len128",
     f"{PLAN2_CELLS}/A1_len128__0000_0500.jsonl",
     "completion", "plan2", "A", 128, 1),
    ("plan2.A2_len256",
     f"{PLAN2_CELLS}/A2_len256__0000_0500.jsonl",
     "completion", "plan2", "A", 256, 1),
    ("plan2.A3_len512",
     f"{PLAN2_CELLS}/A3_len512__0000_0500.jsonl",
     "completion", "plan2", "A", 512, 1),
    ("plan2.A4_len1024",
     f"{PLAN2_CELLS}/A4_len1024__0000_0500.jsonl",
     "completion", "plan2", "A", 1024, 1),
    ("plan2.B1_k1",
     f"{PLAN2_CELLS}/B1_k1__0000_0500.jsonl",
     "voted", "plan2", "B", 512, 1),
    ("plan2.B2_k3",
     f"{PLAN2_CELLS}/B2_k3__0000_0500.jsonl",
     "voted", "plan2", "B", 512, 3),
    ("plan2.B3_k5",
     f"{PLAN2_CELLS}/B3_k5__0000_0500.jsonl",
     "voted", "plan2", "B", 512, 5),
    ("plan2.B4_k10",
     f"{PLAN2_CELLS}/B4_k10__0000_0500.jsonl",
     "voted", "plan2", "B", 512, 10),
]

A3_CELL_NAME = "plan2.A3_len512"

# Plan-3 canonical detection regex. DO NOT change mid-analysis. The fallback
# sweep at --tighten changes both the script's pattern and the manifest in
# the same run, so any tightened report is clearly labelled as such.
CANONICAL_REP_RE = r"(.{10,60})\1{2,}"
TIGHTER_REP_RE = r"(.{15,80})\1{2,}"

OUTCOME_LABELS = ("correct", "terminated_wrong", "repetition_loop",
                  "truncated_no_answer")


# ---- args --------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--results-dir", default=RESULTS_SUBDIR,
        help=f"Directory for analysis outputs. Default {RESULTS_SUBDIR}.",
    )
    p.add_argument(
        "--plan1-cells", default=PLAN1_CELLS,
        help=f"Directory holding plan 1 JSONL shards. Default {PLAN1_CELLS}.",
    )
    p.add_argument(
        "--plan2-cells", default=PLAN2_CELLS,
        help=f"Directory holding plan 2 JSONL shards. Default {PLAN2_CELLS}.",
    )
    p.add_argument(
        "--rep-regex", default=CANONICAL_REP_RE,
        help=("Repetition-detection regex. Default is the plan-3 canonical "
              f"pattern {CANONICAL_REP_RE!r}; the script records this in the "
              "manifest. Use --tighten as a shortcut to the alternative "
              f"{TIGHTER_REP_RE!r}, intended only when the 30-completion "
              "manual sample shows >20%% false positives."),
    )
    p.add_argument(
        "--tighten", action="store_true",
        help=f"Shortcut for --rep-regex {TIGHTER_REP_RE!r}.",
    )
    p.add_argument(
        "--sample-size", type=int, default=30,
        help="Number of flagged completions to dump for manual FP review. "
             "Plan 3 specifies 30.",
    )
    p.add_argument(
        "--sample-seed", type=int, default=42,
        help="RNG seed for the FP-sample selection. Determinism matters so "
             "rerunning the analysis produces an identical sample.",
    )
    p.add_argument(
        "--sticky-top-k", type=int, default=10,
        help="Number of top sticky problem indices to surface (step 6).",
    )
    args = p.parse_args()
    if args.tighten and args.rep_regex == CANONICAL_REP_RE:
        args.rep_regex = TIGHTER_REP_RE
    return args


# ---- IO ----------------------------------------------------------------------

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: malformed JSONL at {path}:{i} ({e}); skipping.")
    rows.sort(key=lambda r: r.get("idx", -1))
    return rows


def regex_hash(pattern):
    return hashlib.sha256(pattern.encode()).hexdigest()[:12]


# ---- classification ----------------------------------------------------------

def make_is_rep_loop(pattern):
    rep_re = re.compile(pattern, re.DOTALL)
    def is_rep_loop(text):
        if not text:
            return False
        return rep_re.search(text) is not None
    return is_rep_loop


def classify_one(completion, pred, gold, is_rep_loop):
    """Mutually-exclusive priority: correct beats rep_loop beats truncated
    beats terminated_wrong. A correct answer reached after some incidental
    repetition is still 'correct' --- rep_loop only labels failures."""
    if pred is not None and pred == gold:
        return "correct"
    if is_rep_loop(completion):
        return "repetition_loop"
    if pred is None:
        return "truncated_no_answer"
    return "terminated_wrong"


def classify_voted(row, is_rep_loop):
    """Problem-level outcome for axis B: voted-correct stays correct;
    otherwise label rep_loop iff the majority of chains rep-looped, then
    fall through to truncated/terminated based on voted_pred."""
    if row.get("correct"):
        return "correct"
    chains = row.get("chains", [])
    if chains:
        n_rep = sum(1 for ch in chains
                    if is_rep_loop(ch.get("completion", "")))
        if n_rep > len(chains) / 2:
            return "repetition_loop"
    if row.get("voted_pred") is None:
        return "truncated_no_answer"
    return "terminated_wrong"


# ---- aggregation -------------------------------------------------------------

def per_cell_completion_counts(rows, is_rep_loop):
    """For 'completion' kind cells: classify each row's completion."""
    counts = Counter({lbl: 0 for lbl in OUTCOME_LABELS})
    rep_loop_idxs = []
    correct_idxs = []
    flagged_completions = []
    for r in rows:
        label = classify_one(r.get("completion", ""), r.get("pred"),
                             r["gold"], is_rep_loop)
        counts[label] += 1
        if label == "repetition_loop":
            rep_loop_idxs.append(r["idx"])
            flagged_completions.append({
                "idx": r["idx"],
                "gold": r["gold"],
                "pred": r.get("pred"),
                "completion": r.get("completion", ""),
            })
        if label == "correct":
            correct_idxs.append(r["idx"])
    return counts, rep_loop_idxs, correct_idxs, flagged_completions


def per_cell_voted_counts(rows, is_rep_loop):
    """Problem-level outcome counts for axis B (voted)."""
    counts = Counter({lbl: 0 for lbl in OUTCOME_LABELS})
    rep_loop_idxs = []
    correct_idxs = []
    for r in rows:
        label = classify_voted(r, is_rep_loop)
        counts[label] += 1
        if label == "repetition_loop":
            rep_loop_idxs.append(r["idx"])
        if label == "correct":
            correct_idxs.append(r["idx"])
    return counts, rep_loop_idxs, correct_idxs


def per_cell_chain_counts(rows, is_rep_loop):
    """Chain-level outcome counts for axis B: aggregate over k * n chains."""
    counts = Counter({lbl: 0 for lbl in OUTCOME_LABELS})
    rep_loop_chain_idxs = []  # (problem_idx, chain_pos)
    flagged_completions = []
    n_chains = 0
    for r in rows:
        gold = r["gold"]
        for j, ch in enumerate(r.get("chains", [])):
            n_chains += 1
            label = classify_one(ch.get("completion", ""), ch.get("pred"),
                                 gold, is_rep_loop)
            counts[label] += 1
            if label == "repetition_loop":
                rep_loop_chain_idxs.append((r["idx"], j))
                flagged_completions.append({
                    "idx": r["idx"],
                    "chain": j,
                    "gold": gold,
                    "pred": ch.get("pred"),
                    "completion": ch.get("completion", ""),
                })
    return counts, n_chains, rep_loop_chain_idxs, flagged_completions


# ---- adjusted accuracies (step 2) -------------------------------------------

def adjusted_accuracies(counts, n, rep_idxs, a3_correct_idxs):
    """Lenient: correct / (n - rep_loop). Upper-bound: imputes A3's outcome
    on this cell's rep-loop indices --- so a rep-loop problem is treated as
    correct iff A3 got it right. For A3 itself the imputation is a no-op."""
    correct = counts["correct"]
    rep = counts["repetition_loop"]
    headline = correct / n if n > 0 else 0.0
    if (n - rep) > 0:
        lenient = correct / (n - rep)
    else:
        lenient = 0.0
    a3_set = set(a3_correct_idxs)
    extras = sum(1 for idx in rep_idxs if idx in a3_set)
    upper = (correct + extras) / n if n > 0 else 0.0
    return {
        "headline": headline,
        "lenient": lenient,
        "upper_bound_a3_imputed": upper,
        "n_rep_loop_problems": rep,
        "n_rep_loop_problems_a3_correct": extras,
    }


# ---- step 5: vote-on-repetition ---------------------------------------------

def vote_on_rep_breakdown(rows, is_rep_loop):
    """For axis B: among voted-wrong problems, classify each by how
    rep-loops were distributed across the k chains. Returns the absolute
    counts; the printer turns them into rates over the voted-wrong total."""
    cats = Counter({
        "unanimous_rep_loop": 0,
        "rep_loop_plurality_wrong": 0,
        "rep_loop_unlucky": 0,
        "no_rep_loop_wrong": 0,
        "other": 0,
    })
    voted_wrong = 0
    examples = defaultdict(list)
    for r in rows:
        if r.get("correct"):
            continue
        voted_wrong += 1
        gold = r["gold"]
        chains = r.get("chains", [])
        k = len(chains)
        if k == 0:
            cats["other"] += 1
            continue
        rep_chain = [is_rep_loop(ch.get("completion", "")) for ch in chains]
        chain_pred = [ch.get("pred") for ch in chains]
        R = sum(rep_chain)
        C = sum(1 for j in range(k)
                if not rep_chain[j] and chain_pred[j] == gold)
        W = sum(1 for j in range(k)
                if not rep_chain[j] and chain_pred[j] is not None
                and chain_pred[j] != gold)
        # T (truncated chains) = k - R - C - W; not directly used in
        # the categorisation but useful to have for examples below.
        if R == k:
            cat = "unanimous_rep_loop"
        elif R > k / 2 and C >= 1:
            cat = "rep_loop_plurality_wrong"
        elif R == 0:
            cat = "no_rep_loop_wrong"
        elif R < k / 2 and W > R:
            cat = "rep_loop_unlucky"
        else:
            cat = "other"
        cats[cat] += 1
        if len(examples[cat]) < 3:
            examples[cat].append({"idx": r["idx"], "gold": gold,
                                  "k": k, "R": R, "C": C, "W": W,
                                  "voted_pred": r.get("voted_pred")})
    return {"voted_wrong": voted_wrong,
            "categories": dict(cats),
            "examples": {k: v for k, v in examples.items()}}


# ---- step 6: sticky problems -------------------------------------------------

def sticky_problems(per_cell_rep_loop_idxs, per_cell_n, top_k):
    """Count how many cells flagged each idx. We weight every cell equally
    and report ``triggered / cells_covering_idx`` so plan-1 cells (which
    only cover idx 0..99) don't unfairly dominate via missing data."""
    coverage = defaultdict(int)
    triggered = defaultdict(int)
    triggered_by = defaultdict(list)
    for cell_name, _ in per_cell_n.items():
        max_idx = per_cell_n[cell_name]
        for idx in range(max_idx):
            coverage[idx] += 1
        for idx in per_cell_rep_loop_idxs.get(cell_name, []):
            triggered[idx] += 1
            triggered_by[idx].append(cell_name)
    rows = []
    for idx, cnt in triggered.items():
        cov = coverage.get(idx, 1)
        rows.append((idx, cnt, cov, cnt / cov, sorted(triggered_by[idx])))
    rows.sort(key=lambda x: (-x[1], -x[3], x[0]))
    return [{"idx": idx, "n_cells_triggered": cnt,
             "n_cells_covering": cov, "rate": rate,
             "cells": cells}
            for idx, cnt, cov, rate, cells in rows[:top_k]]


def gsm8k_problem_text(idxs):
    """Look up the GSM8K-test question for the sticky-problem table.
    Skipped silently if datasets isn't importable --- the rest of the
    analysis doesn't depend on it."""
    if not idxs:
        return {}
    try:
        from datasets import load_dataset
    except ImportError:
        print("  (datasets not installed; sticky problem text unavailable)")
        return {}
    try:
        gsm = load_dataset("gsm8k", "main", split="test")
    except Exception as e:
        print(f"  (could not load GSM8K: {e}; sticky problem text unavailable)")
        return {}
    out = {}
    for idx in idxs:
        if 0 <= idx < len(gsm):
            row = gsm[int(idx)]
            out[int(idx)] = {"question": row["question"],
                             "answer": row["answer"]}
    return out


# ---- false-positive sample heuristic ----------------------------------------

def heuristic_repetition_count(completion, pattern):
    """Return the largest contiguous repeat count of any captured substring
    matched by `pattern` in `completion`. Used to flag obvious true positives
    (rep_count >> 3) versus borderline cases for the manual FP sample."""
    rep_re = re.compile(pattern, re.DOTALL)
    best = 0
    for m in rep_re.finditer(completion):
        capt = m.group(1)
        if not capt:
            continue
        # The pattern matches \1{2,} after the capture; count the actual
        # repeats of the capture starting at the capture position.
        start = m.start(1)
        n = 0
        L = len(capt)
        while completion[start + n * L : start + (n + 1) * L] == capt:
            n += 1
        if n > best:
            best = n
    return best


def fp_sample_dump(all_flagged, args, results_dir):
    """Pick `args.sample_size` flagged completions uniformly at random across
    all cells (deterministic via --sample-seed) and dump them for manual FP
    review. Also computes a heuristic-based auto FP estimate (entries with
    rep_count <= 3 are 'borderline'; >= 5 are 'definite TP') as a directional
    signal that is NOT a substitute for the manual review."""
    rng = random.Random(args.sample_seed)
    if not all_flagged:
        return None, None
    pool = list(all_flagged)
    rng.shuffle(pool)
    sample = pool[: args.sample_size]
    enriched = []
    borderline = 0
    definite_tp = 0
    for entry in sample:
        rep_count = heuristic_repetition_count(entry["completion"],
                                               args.rep_regex)
        # Heuristic: flag low-repeat captures whose surrounding text contains
        # large amounts of varied (non-matched) content as "borderline".
        # Rep count 5+ is taken as a definite true positive (5+ verbatim
        # repeats of a 10+ char span is overwhelmingly degenerate output).
        kind = "definite_tp" if rep_count >= 5 else (
            "borderline" if rep_count <= 3 else "likely_tp"
        )
        if kind == "borderline":
            borderline += 1
        if kind == "definite_tp":
            definite_tp += 1
        enriched.append({**entry,
                         "max_repetition_count": rep_count,
                         "heuristic_label": kind})
    out = {
        "sample_size": len(sample),
        "regex": args.rep_regex,
        "regex_hash": regex_hash(args.rep_regex),
        "seed": args.sample_seed,
        "heuristic_summary": {
            "definite_tp_count": definite_tp,
            "borderline_count": borderline,
            "borderline_rate": borderline / len(sample) if sample else 0.0,
            "note": ("borderline = rep_count <= 3 (the regex's minimum). "
                     "definite_tp = rep_count >= 5. The plan-3 false-"
                     "positive threshold is >20%; if the heuristic "
                     "borderline_rate exceeds that, the user should "
                     "manually inspect this file and consider --tighten."),
        },
        "samples": enriched,
    }
    out_path = Path(results_dir) / "flagged_sample_30.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    return out_path, out["heuristic_summary"]


# ---- pre-registered finding letter ------------------------------------------

def determine_finding(per_cell_results, axis_b_chain_rates, a3_chain_rate=None):
    """Walk plan-3's pre-registered thresholds and return one of A/B/C/D/E.
    The letter is informational: the writeup uses the actual numbers, but
    surfacing the canonical label keeps the pre-registration honest."""
    # Findings A/B/C compare adjusted accuracies (lenient and upper-bound)
    # against headline accuracy across all cells. Use the *largest* gap
    # observed across cells to set the global label, since the plan asks
    # "does the change vary by cell?" --- if any cell shifts >8 points,
    # the conservative answer is C.
    max_gap = 0.0
    for cell, info in per_cell_results.items():
        if info is None:
            continue
        h = info["headline"]
        gap = max(info["lenient"] - h, info["upper_bound_a3_imputed"] - h)
        if gap > max_gap:
            max_gap = gap
    if max_gap < 0.03:
        primary = ("A", "Rep-loops are a small factor (<3 pts adjusted-accuracy "
                   "swing in any cell). Path 1's ceiling is genuinely a "
                   "reasoning ceiling.")
    elif max_gap < 0.08:
        primary = ("B", "Rep-loops are a medium factor (3-8 pts adjusted-"
                   "accuracy swing). Path 1's ceiling is part reasoning, part "
                   "stability; Paths 2-4 could pick up stability points.")
    else:
        primary = ("C", f"Rep-loops are a large factor (>8 pts swing in some "
                   f"cell, max gap {max_gap:.3f}). The plan-2 ceiling is "
                   f"significantly inflated by stability failures; head-to-"
                   f"head needs a generation-stability metric.")
    # D/E are a SECONDARY signal about axis-B chain-level vs A3 greedy, only
    # applicable if A3's rate is known and there is at least one B cell.
    secondary = None
    if a3_chain_rate is not None and axis_b_chain_rates:
        b_rates = list(axis_b_chain_rates.values())
        max_b = max(b_rates)
        min_b = min(b_rates)
        # Use a 1.5x ratio as the "<<" / ">>" threshold (plan is qualitative).
        if max_b > 1.5 * a3_chain_rate:
            secondary = ("D", f"Axis-B chain-level rep-loop rate (max "
                         f"{max_b:.3f}) >> A3's ({a3_chain_rate:.3f}). "
                         f"Sampled chains are markedly more rep-loop-prone "
                         f"than greedy --- mechanism behind plan-2 SC-FLAT.")
        elif min_b < (a3_chain_rate / 1.5) and a3_chain_rate > 0:
            secondary = ("E", f"Axis-B chain-level rep-loop rate (min "
                         f"{min_b:.3f}) << A3's ({a3_chain_rate:.3f}). "
                         f"Sampling is implicitly escaping rep-loop "
                         f"attractors --- a free stability win for Path 4.")
    return primary, secondary


# ---- printing ---------------------------------------------------------------

def fmt_pct(x):
    return f"{100 * x:5.1f}%"


def print_outcome_table(per_cell_outcomes, per_cell_chain_outcomes):
    print()
    print(f"{'cell':24s}  {'n':>5s}  {'correct':>9s}  {'term_wrong':>10s}"
          f"  {'rep_loop':>9s}  {'trunc':>7s}")
    print("-" * 76)
    for cell_name, _, kind, *_ in CELLS:
        info = per_cell_outcomes.get(cell_name)
        if info is None:
            continue
        n = info["n"]
        c = info["counts"]
        print(f"{cell_name:24s}  {n:5d}  "
              f"{c['correct']:5d} ({fmt_pct(c['correct']/n)})"
              f"  {c['terminated_wrong']:5d} ({fmt_pct(c['terminated_wrong']/n)})"
              f"  {c['repetition_loop']:5d} ({fmt_pct(c['repetition_loop']/n)})"
              f"  {c['truncated_no_answer']:3d} ({fmt_pct(c['truncated_no_answer']/n)})")
        if kind == "voted":
            chain = per_cell_chain_outcomes.get(cell_name)
            if chain is not None:
                cn = chain["n_chains"]
                cc = chain["counts"]
                cell_chain_lbl = f"  {cell_name} (chain)"
                print(f"{cell_chain_lbl:24s}  {cn:5d}  "
                      f"{cc['correct']:5d} ({fmt_pct(cc['correct']/cn)})"
                      f"  {cc['terminated_wrong']:5d} ({fmt_pct(cc['terminated_wrong']/cn)})"
                      f"  {cc['repetition_loop']:5d} ({fmt_pct(cc['repetition_loop']/cn)})"
                      f"  {cc['truncated_no_answer']:3d} ({fmt_pct(cc['truncated_no_answer']/cn)})")


def print_adjusted_table(per_cell_adjusted):
    print()
    print(f"{'cell':24s}  {'headline':>9s}  {'lenient':>8s}  "
          f"{'upper(A3)':>10s}  {'lenient-h':>10s}  {'upper-h':>9s}")
    print("-" * 78)
    for cell_name, _, *_ in CELLS:
        adj = per_cell_adjusted.get(cell_name)
        if adj is None:
            continue
        h = adj["headline"]
        l = adj["lenient"]
        u = adj["upper_bound_a3_imputed"]
        print(f"{cell_name:24s}  {fmt_pct(h)}    {fmt_pct(l)}     {fmt_pct(u)}      "
              f"{(l - h) * 100:+5.1f} pp     {(u - h) * 100:+5.1f} pp")


def print_length_table(axis_a_rep_rates):
    print()
    print("Length-dependence (axis A, greedy):")
    print(f"{'cell':16s}  {'max_new':>7s}  {'rep_loop_rate':>13s}")
    for cell, max_new, rate in axis_a_rep_rates:
        print(f"{cell:16s}  {max_new:7d}  {fmt_pct(rate)}")


def print_sampling_table(a3_chain_rate, axis_b_chain_rates):
    print()
    print("Sampling-dependence (axis B chains vs A3 greedy):")
    print(f"{'cell':16s}  {'k':>3s}  {'chain_rep_rate':>14s}  {'vs_A3':>7s}")
    print(f"{A3_CELL_NAME:16s}  {1:3d}  {fmt_pct(a3_chain_rate):>14s}  {'(ref)':>7s}")
    for cell, k, rate in axis_b_chain_rates:
        if a3_chain_rate > 0:
            ratio = rate / a3_chain_rate
            ratio_s = f"{ratio:.2f}x"
        else:
            ratio_s = "n/a"
        print(f"{cell:16s}  {k:3d}  {fmt_pct(rate):>14s}  {ratio_s:>7s}")


def print_vote_breakdown(axis_b_vote_breakdowns):
    print()
    print("Vote-on-repetition breakdown (axis B, voted-wrong problems only):")
    cats = ("unanimous_rep_loop", "rep_loop_plurality_wrong",
            "rep_loop_unlucky", "no_rep_loop_wrong", "other")
    header = f"{'cell':12s}  {'voted_wrong':>11s}"
    for c in cats:
        header += f"  {c:>23s}"
    print(header)
    for cell, info in axis_b_vote_breakdowns.items():
        vw = info["voted_wrong"]
        line = f"{cell:12s}  {vw:11d}"
        for c in cats:
            n = info["categories"].get(c, 0)
            rate = (n / vw) if vw else 0.0
            line += f"  {n:5d} ({fmt_pct(rate):>6s})        "
        print(line)


def print_sticky(sticky, gsm_text):
    print()
    print("Top sticky problems (most cells flagging the same idx as rep_loop):")
    print(f"{'idx':>4s}  {'n_trig':>6s}  {'cov':>3s}  {'rate':>5s}  cells")
    for s in sticky:
        cells_short = ",".join(c.replace("plan2.", "").replace("plan1.", "p1.")
                               for c in s["cells"])
        print(f"{s['idx']:4d}  {s['n_cells_triggered']:6d}  "
              f"{s['n_cells_covering']:3d}  {fmt_pct(s['rate'])}  {cells_short}")
        q = gsm_text.get(s["idx"], {}).get("question", "")
        if q:
            qs = q[:120].replace("\n", " ")
            print(f"      Q: {qs}{'...' if len(q) > 120 else ''}")


# ---- CSV + plot --------------------------------------------------------------

def write_csv(path, per_cell_outcomes, per_cell_adjusted,
              per_cell_chain_outcomes):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "kind", "axis", "max_new_tokens", "k", "n",
                    "correct", "terminated_wrong", "repetition_loop",
                    "truncated_no_answer", "rep_loop_rate",
                    "headline_acc", "lenient_acc", "upper_bound_acc",
                    "n_chains", "chain_rep_loop_rate"])
        for cell_name, _, kind, _, axis, max_new, k in CELLS:
            o = per_cell_outcomes.get(cell_name)
            a = per_cell_adjusted.get(cell_name)
            if o is None or a is None:
                continue
            n = o["n"]
            counts = o["counts"]
            chain = per_cell_chain_outcomes.get(cell_name)
            n_chains = chain["n_chains"] if chain else ""
            chain_rate = (chain["counts"]["repetition_loop"] / chain["n_chains"]
                          if chain and chain["n_chains"] > 0 else "")
            w.writerow([cell_name, kind, axis, max_new, k, n,
                        counts["correct"], counts["terminated_wrong"],
                        counts["repetition_loop"],
                        counts["truncated_no_answer"],
                        f"{counts['repetition_loop'] / n:.6f}",
                        f"{a['headline']:.6f}",
                        f"{a['lenient']:.6f}",
                        f"{a['upper_bound_a3_imputed']:.6f}",
                        n_chains, chain_rate if chain_rate == ""
                        else f"{chain_rate:.6f}"])


def make_plot(path, per_cell_outcomes, per_cell_adjusted, axis_a_rep_rates,
              axis_b_chain_rates, a3_chain_rate, vote_breakdowns,
              finding_letter):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax1, ax2, ax3, ax4 = axes.flat

    # Panel 1: per-cell rep-loop rate (problem-level).
    cell_names = []
    rep_rates = []
    for cell_name, *_ in CELLS:
        o = per_cell_outcomes.get(cell_name)
        if o is None:
            continue
        cell_names.append(cell_name.replace("plan2.", "").replace("plan1.", ""))
        rep_rates.append(o["counts"]["repetition_loop"] / o["n"])
    xs = list(range(len(cell_names)))
    bars = ax1.bar(xs, rep_rates, color="#c44e4e", edgecolor="black",
                   linewidth=0.6)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(cell_names, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("rep-loop rate (problem-level)")
    ax1.set_title("Per-cell repetition-loop rate")
    ax1.grid(axis="y", linestyle=":", alpha=0.4)
    for x, v in zip(xs, rep_rates):
        ax1.text(x, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    # Panel 2: headline vs lenient vs upper-bound adjusted accuracy.
    headlines, lenients, uppers, names2 = [], [], [], []
    for cell_name, *_ in CELLS:
        a = per_cell_adjusted.get(cell_name)
        if a is None:
            continue
        names2.append(cell_name.replace("plan2.", "").replace("plan1.", ""))
        headlines.append(a["headline"])
        lenients.append(a["lenient"])
        uppers.append(a["upper_bound_a3_imputed"])
    xs2 = list(range(len(names2)))
    width = 0.27
    ax2.bar([x - width for x in xs2], headlines, width, label="headline",
            color="#444444", edgecolor="black", linewidth=0.5)
    ax2.bar(xs2, lenients, width, label="lenient", color="#1f6fb4",
            edgecolor="black", linewidth=0.5)
    ax2.bar([x + width for x in xs2], uppers, width, label="upper(A3)",
            color="#5fa55a", edgecolor="black", linewidth=0.5)
    ax2.set_xticks(xs2)
    ax2.set_xticklabels(names2, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("accuracy")
    ax2.set_title("Adjusted accuracies under counterfactuals")
    ax2.legend(loc="lower right", fontsize=8, frameon=False)
    ax2.grid(axis="y", linestyle=":", alpha=0.4)

    # Panel 3: length-dependence + sampling-dependence on a single axis.
    if axis_a_rep_rates:
        a_lens = [m for _, m, _ in axis_a_rep_rates]
        a_rates = [r for *_, r in axis_a_rep_rates]
        ax3.plot(a_lens, a_rates, "o-", color="#1f6fb4", label="axis A (greedy)",
                 linewidth=1.5)
        for x, y in zip(a_lens, a_rates):
            ax3.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                         xytext=(5, 4), fontsize=7)
    if axis_b_chain_rates:
        # Plot axis B chain-level rates at len=512 (same x as A3) but with
        # a small horizontal offset per k for legibility.
        for i, (cell, k, rate) in enumerate(axis_b_chain_rates):
            ax3.plot([512 + (i + 1) * 12], [rate], "s", color="#c44e4e",
                     markersize=8, label=f"axis B chain k={k}" if i == 0
                     else None)
            ax3.annotate(f"k={k}\n{rate:.2f}",
                         (512 + (i + 1) * 12, rate),
                         textcoords="offset points", xytext=(6, -2),
                         fontsize=6, color="#c44e4e")
    if a3_chain_rate is not None:
        ax3.axhline(a3_chain_rate, linestyle="--", color="#1f6fb4",
                    alpha=0.5, linewidth=0.8)
        ax3.text(140, a3_chain_rate, f"A3 ref ({a3_chain_rate:.2f})",
                 fontsize=7, color="#1f6fb4", va="bottom")
    ax3.set_xlabel("max_new_tokens")
    ax3.set_ylabel("rep-loop rate")
    ax3.set_title("Length-dependence (A) and sampling-dependence (B chains)")
    ax3.grid(True, linestyle=":", alpha=0.4)
    ax3.legend(loc="upper left", fontsize=7, frameon=False)

    # Panel 4: vote-on-rep stacked bar across B cells.
    cats = ("unanimous_rep_loop", "rep_loop_plurality_wrong",
            "rep_loop_unlucky", "no_rep_loop_wrong", "other")
    palette = ["#7d2828", "#c44e4e", "#e8a87c", "#bbbbbb", "#666666"]
    b_names = list(vote_breakdowns.keys())
    if b_names:
        bottoms = [0] * len(b_names)
        for cat, color in zip(cats, palette):
            ys = []
            for name in b_names:
                vw = vote_breakdowns[name]["voted_wrong"]
                ys.append((vote_breakdowns[name]["categories"].get(cat, 0) /
                           vw) if vw else 0.0)
            ax4.bar(b_names, ys, bottom=bottoms, label=cat, color=color,
                    edgecolor="black", linewidth=0.4)
            bottoms = [b + y for b, y in zip(bottoms, ys)]
    ax4.set_ylabel("share of voted-wrong problems")
    ax4.set_title("Vote-on-repetition (axis B, voted-wrong only)")
    ax4.legend(loc="upper right", fontsize=7, frameon=False)
    ax4.grid(axis="y", linestyle=":", alpha=0.4)
    ax4.tick_params(axis="x", rotation=30)

    fig.suptitle(f"Path 1 plan 3 --- repetition-loop isolation  "
                 f"(finding {finding_letter[0]})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


# ---- main --------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"plan-3 rep-loop analysis  regex={args.rep_regex!r}  "
          f"hash={regex_hash(args.rep_regex)}")
    is_rep_loop = make_is_rep_loop(args.rep_regex)

    # First pass: load every cell and classify problem-level outcomes.
    rows_by_cell = {}
    per_cell_outcomes = {}
    per_cell_chain_outcomes = {}
    rep_loop_idxs_by_cell = {}
    correct_idxs_by_cell = {}
    flagged_for_sample = []

    for cell_name, path, kind, _, _, _, _ in CELLS:
        p = Path(path)
        if not p.exists():
            print(f"WARNING: {p} missing; skipping cell {cell_name}.")
            continue
        rows = load_jsonl(p)
        rows_by_cell[cell_name] = rows
        if kind == "completion":
            counts, rep_idxs, correct_idxs, flagged = (
                per_cell_completion_counts(rows, is_rep_loop))
            per_cell_outcomes[cell_name] = {"n": len(rows), "counts": dict(counts)}
            rep_loop_idxs_by_cell[cell_name] = rep_idxs
            correct_idxs_by_cell[cell_name] = correct_idxs
            for entry in flagged:
                flagged_for_sample.append({"cell": cell_name, **entry})
        elif kind == "voted":
            counts, rep_idxs, correct_idxs = per_cell_voted_counts(
                rows, is_rep_loop)
            per_cell_outcomes[cell_name] = {"n": len(rows), "counts": dict(counts)}
            rep_loop_idxs_by_cell[cell_name] = rep_idxs
            correct_idxs_by_cell[cell_name] = correct_idxs
            chain_counts, n_chains, _, chain_flagged = (
                per_cell_chain_counts(rows, is_rep_loop))
            per_cell_chain_outcomes[cell_name] = {
                "n_chains": n_chains, "counts": dict(chain_counts),
            }
            for entry in chain_flagged:
                flagged_for_sample.append({"cell": cell_name, **entry})

    # A3 correct-idx set is the substrate for the upper-bound counterfactual.
    a3_correct = correct_idxs_by_cell.get(A3_CELL_NAME, [])

    # Step 2: adjusted accuracies.
    per_cell_adjusted = {}
    for cell_name, _, kind, *_ in CELLS:
        info = per_cell_outcomes.get(cell_name)
        if info is None:
            continue
        adj = adjusted_accuracies(info["counts"], info["n"],
                                  rep_loop_idxs_by_cell[cell_name],
                                  a3_correct)
        per_cell_adjusted[cell_name] = adj

    # Step 3: length-dependence (axis A only).
    axis_a_rep_rates = []
    for cell_name, _, kind, _, axis, max_new, _ in CELLS:
        if axis != "A":
            continue
        o = per_cell_outcomes.get(cell_name)
        if o is None:
            continue
        rate = o["counts"]["repetition_loop"] / o["n"]
        axis_a_rep_rates.append((cell_name, max_new, rate))

    # Step 4: sampling-dependence (axis B chain-level vs A3 greedy).
    a3_outcome = per_cell_outcomes.get(A3_CELL_NAME)
    a3_rep_rate = (a3_outcome["counts"]["repetition_loop"] / a3_outcome["n"]
                   if a3_outcome else None)
    axis_b_chain_rates = []
    axis_b_chain_rate_dict = {}
    for cell_name, _, kind, _, axis, _, k in CELLS:
        if axis != "B":
            continue
        chain = per_cell_chain_outcomes.get(cell_name)
        if chain is None:
            continue
        rate = (chain["counts"]["repetition_loop"] / chain["n_chains"]
                if chain["n_chains"] > 0 else 0.0)
        axis_b_chain_rates.append((cell_name, k, rate))
        axis_b_chain_rate_dict[cell_name] = rate

    # Step 5: vote-on-rep breakdown for axis B.
    vote_breakdowns = {}
    for cell_name, _, kind, _, axis, _, _ in CELLS:
        if axis != "B":
            continue
        rows = rows_by_cell.get(cell_name)
        if rows is None:
            continue
        vote_breakdowns[cell_name.replace("plan2.", "")] = (
            vote_on_rep_breakdown(rows, is_rep_loop))

    # Step 6: sticky problems.
    per_cell_n = {cell_name: per_cell_outcomes[cell_name]["n"]
                  for cell_name, *_ in CELLS
                  if cell_name in per_cell_outcomes}
    sticky = sticky_problems(rep_loop_idxs_by_cell, per_cell_n,
                             args.sticky_top_k)
    gsm_text = gsm8k_problem_text([s["idx"] for s in sticky])

    # Pre-registered finding label.
    primary, secondary = determine_finding(per_cell_adjusted,
                                           axis_b_chain_rate_dict,
                                           a3_rep_rate)

    # Print human-readable summary.
    print_outcome_table(per_cell_outcomes, per_cell_chain_outcomes)
    print_adjusted_table(per_cell_adjusted)
    print_length_table(axis_a_rep_rates)
    if a3_rep_rate is not None:
        print_sampling_table(a3_rep_rate, axis_b_chain_rates)
    print_vote_breakdown(vote_breakdowns)
    print_sticky(sticky, gsm_text)

    # FP-sample dump (manual inspection target).
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    sample_path, sample_summary = fp_sample_dump(flagged_for_sample, args,
                                                 results_dir)
    print()
    if sample_path is not None:
        print(f"FP-sample (n={args.sample_size}, manual review) -> {sample_path}")
        print(f"  heuristic borderline_rate = "
              f"{fmt_pct(sample_summary['borderline_rate'])}  "
              f"(definite_tp={sample_summary['definite_tp_count']}, "
              f"borderline={sample_summary['borderline_count']})")
        if sample_summary["borderline_rate"] > 0.20:
            print("  WARNING: heuristic borderline_rate > 20%. Manually "
                  "review the sample; if confirmed, rerun with --tighten.")
    else:
        print("FP-sample: no rep-loop completions to sample from.")

    # Outputs.
    out = {
        "config": {
            "regex": args.rep_regex,
            "regex_hash": regex_hash(args.rep_regex),
            "regex_canonical": CANONICAL_REP_RE,
            "tightened": args.rep_regex != CANONICAL_REP_RE,
            "plan1_cells": str(args.plan1_cells),
            "plan2_cells": str(args.plan2_cells),
            "sample_size": args.sample_size,
            "sample_seed": args.sample_seed,
            "sticky_top_k": args.sticky_top_k,
            "limitations": (
                "Regex catches verbatim repeats of 10-60 char spans only. "
                "False positives: legitimate arithmetic restatements with "
                "variation will not trigger; a 30-completion manual sample "
                "is dumped to flagged_sample_30.json so the writeup can "
                "report the FP rate from inspection. False negatives: "
                "non-verbatim semantic repetitions are missed (likely "
                "10-20% undercount). English-only; not an issue for GSM8K."
            ),
        },
        "per_cell_outcomes": per_cell_outcomes,
        "per_cell_chain_outcomes": per_cell_chain_outcomes,
        "per_cell_adjusted_accuracy": per_cell_adjusted,
        "length_dependence": [
            {"cell": c, "max_new_tokens": m, "rep_loop_rate": r}
            for c, m, r in axis_a_rep_rates],
        "sampling_dependence": {
            "a3_chain_rep_rate": a3_rep_rate,
            "axis_b_chain_rep_rates": [
                {"cell": c, "k": k, "rep_loop_rate": r}
                for c, k, r in axis_b_chain_rates],
        },
        "vote_on_repetition": vote_breakdowns,
        "sticky_problems": [{**s, "question_text":
                             gsm_text.get(s["idx"], {}).get("question"),
                             "gold_answer":
                             gsm_text.get(s["idx"], {}).get("answer")}
                            for s in sticky],
        "fp_sample_summary": sample_summary,
        "finding": {
            "letter": primary[0],
            "primary_reason": primary[1],
            "secondary_letter": secondary[0] if secondary else None,
            "secondary_reason": secondary[1] if secondary else None,
        },
    }
    json_path = results_dir / "results_plan3.json"
    json_path.write_text(json.dumps(out, indent=2))
    print()
    print(f"wrote {json_path}")

    csv_path = results_dir / "path1_rep_loops_by_cell.csv"
    write_csv(csv_path, per_cell_outcomes, per_cell_adjusted,
              per_cell_chain_outcomes)
    print(f"wrote {csv_path}")

    plot_path = results_dir / "results_plan3.png"
    plot_returned = make_plot(plot_path, per_cell_outcomes, per_cell_adjusted,
                              axis_a_rep_rates, axis_b_chain_rates,
                              a3_rep_rate, vote_breakdowns, primary)
    if plot_returned is not None:
        print(f"wrote {plot_returned}")

    print()
    print(f"FINDING (primary): {primary[0]} --- {primary[1]}")
    if secondary:
        print(f"FINDING (axis B): {secondary[0]} --- {secondary[1]}")


if __name__ == "__main__":
    main()
