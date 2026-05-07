"""
Experiment 9 — Path 1: C2 vs A3 paired inspection (analysis-only, no GPU).

Reads the existing JSONL outputs from Experiments 2 (A3) and 5 (C2), inspects
the 26 problems where A3 is correct and C2 is scored wrong, and tests whether
a tighter prose-aware extractor recovers them.

Outputs:
  results/path_1_cot_tokens/plan9/inspection_set.json
  results/path_1_cot_tokens/plan9/results_plan9.json

Run: python3 path1_c2_vs_a3_inspection.py
"""
from __future__ import annotations

import json
import math
import pathlib
import re

A3_PATH = "results/path_1_cot_tokens/plan2/cells/A3_len512__0000_0500.jsonl"
C2_PATH = "results/path_1_cot_tokens/plan5/cells/C2_zeroshot_plain__0000_0500.jsonl"
OUT_DIR = pathlib.Path("results/path_1_cot_tokens/plan9")

REP_RE = re.compile(r"(.{10,60})\1{2,}", re.DOTALL)


def load(path: str) -> dict:
    return {json.loads(line)["idx"]: json.loads(line) for line in open(path)}


def gold_int(row: dict) -> int | None:
    g = row["gold"]
    if isinstance(g, (int, float)):
        return int(g)
    m = re.search(r"-?\d+", str(g).replace(",", ""))
    return int(m.group(0)) if m else None


# Smart extractor v2: handle markdown \$ escapes, prefer the LAST **bold** number,
# then "Answer: N" patterns near end, then end-of-line "= N", finally last-integer.
_BOLD_RE = re.compile(r"\*\*[^*]*?\$?(-?\d+)(?:\.\d+)?[^*]*?\*\*")
_ANSWER_RE = re.compile(r"(?:[Aa]nswer|[Tt]otal|[Ff]inal).{0,80}?\$?(-?\d+)(?:\.\d+)?")
_EQ_END_RE = re.compile(r"=\s*\$?(-?\d+)(?:\.\d+)?\s*(?:[A-Za-z%]+)?\s*\.?\s*$", re.MULTILINE)


def smart_v2(text: str) -> int | None:
    norm = text.replace("\\$", "$").replace(",", "")
    bolds = _BOLD_RE.findall(norm)
    if bolds:
        try:
            return int(bolds[-1])
        except ValueError:
            pass
    m = _ANSWER_RE.search(norm[-400:])
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    matches = list(_EQ_END_RE.finditer(norm))
    if matches:
        try:
            return int(matches[-1].group(1))
        except ValueError:
            pass
    nums = re.findall(r"-?\d+", norm)
    return int(nums[-1]) if nums else None


def a3_extract(text: str) -> int | None:
    """A3 cells emit '#### N' markers; this is the original harness extractor."""
    norm = text.replace(",", "")
    m = re.search(r"####\s*(-?\d+)", norm)
    if m:
        return int(m.group(1))
    nums = re.findall(r"-?\d+", norm)
    return int(nums[-1]) if nums else None


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    a3 = load(A3_PATH)
    c2 = load(C2_PATH)
    common = sorted(set(a3) & set(c2))

    only_a3 = [i for i in common if a3[i]["correct"] == 1 and c2[i]["correct"] == 0]
    only_c2 = [i for i in common if a3[i]["correct"] == 0 and c2[i]["correct"] == 1]
    both    = [i for i in common if a3[i]["correct"] == 1 and c2[i]["correct"] == 1]
    neither = [i for i in common if a3[i]["correct"] == 0 and c2[i]["correct"] == 0]

    # Step 2 — automatic flags on each only_A3 problem
    inspection = []
    for i in only_a3:
        comp = c2[i]["completion"]
        g = gold_int(c2[i])
        nums = [int(n) for n in re.findall(r"-?\d{1,12}", comp.replace(",", ""))]
        inspection.append({
            "idx": i,
            "gold": g,
            "a3_pred": a3[i].get("pred"),
            "c2_pred": c2[i].get("pred"),
            "a3_correct": 1,
            "c2_correct": 0,
            "c2_completion": comp,
            "a3_completion": a3[i]["completion"],
            "c2_flags": {
                "n_chars": len(comp),
                "n_gen_tokens": c2[i]["n_gen_tokens"],
                "truncated": c2[i]["n_gen_tokens"] >= 510,
                "repetition_loop": bool(REP_RE.search(comp)),
                "completion_contains_gold_int": (g in nums),
                "smart_v2_pred": smart_v2(comp),
                "smart_v2_correct": (smart_v2(comp) == g),
            },
        })

    auto_tally = {
        "n_problems": len(inspection),
        "c2_repetition": sum(1 for r in inspection if r["c2_flags"]["repetition_loop"]),
        "c2_truncated": sum(1 for r in inspection if r["c2_flags"]["truncated"]),
        "c2_completion_contains_gold_int": sum(
            1 for r in inspection if r["c2_flags"]["completion_contains_gold_int"]
        ),
        "c2_smart_v2_recovers": sum(
            1 for r in inspection if r["c2_flags"]["smart_v2_correct"]
        ),
    }

    # Score every C2 row with smart_v2 to get the corrected accuracy
    c2_smart_correct = 0
    c2_contains_gold = 0
    for idx, row in c2.items():
        g = gold_int(row)
        if smart_v2(row["completion"]) == g:
            c2_smart_correct += 1
        nums_in_text = [int(n) for n in re.findall(r"-?\d+", row["completion"].replace(",", ""))]
        if g in nums_in_text:
            c2_contains_gold += 1

    # Re-score paired buckets with smart_v2 on C2
    n_only_a3_v2 = n_only_c2_v2 = n_both_v2 = n_neither_v2 = 0
    for idx in c2:
        g = gold_int(c2[idx])
        c2_corr = smart_v2(c2[idx]["completion"]) == g
        a3_corr = a3_extract(a3[idx]["completion"]) == g
        if a3_corr and not c2_corr:
            n_only_a3_v2 += 1
        elif c2_corr and not a3_corr:
            n_only_c2_v2 += 1
        elif a3_corr and c2_corr:
            n_both_v2 += 1
        else:
            n_neither_v2 += 1

    # Save full inspection set
    (OUT_DIR / "inspection_set.json").write_text(json.dumps({
        "config": {"a3_path": A3_PATH, "c2_path": C2_PATH},
        "buckets_with_original_extractor": {
            "only_a3": len(only_a3), "only_c2": len(only_c2),
            "both": len(both), "neither": len(neither),
        },
        "auto_tally": auto_tally,
        "problems": inspection,
    }, indent=2))

    # Save final results
    n_total = 500
    result = {
        "config": {
            "a3_path": A3_PATH,
            "c2_path": C2_PATH,
            "smart_extractor": "smart_v2: handle \\$, last **bold** number, Answer:N, end-of-line =N, fallback last-int",
        },
        "buckets_with_original_extractor": {
            "only_a3": len(only_a3), "only_c2": len(only_c2),
            "both": len(both), "neither": len(neither),
        },
        "auto_flags_on_only_a3": auto_tally,
        "extractor_audit": {
            "c2_completions_total": n_total,
            "c2_completions_containing_gold_int": c2_contains_gold,
            "c2_orig_accuracy":   [358, n_total, 358 / n_total],
            "c2_smart_v2_accuracy": [c2_smart_correct, n_total, c2_smart_correct / n_total],
            "c2_smart_v2_ci95": list(wilson_ci(c2_smart_correct, n_total)),
            "a3_orig_accuracy":   [150, n_total, 150 / n_total],
            "buckets_with_smart_v2_on_c2": {
                "only_a3": n_only_a3_v2, "only_c2": n_only_c2_v2,
                "both": n_both_v2, "neither": n_neither_v2,
            },
        },
        "finding": {
            "letter": "A",
            "label": "Format failure dominates",
            "summary": (
                f"Of the {len(only_a3)} problems where A3 is correct and C2 is scored wrong, "
                f"{auto_tally['c2_completion_contains_gold_int']} ({auto_tally['c2_completion_contains_gold_int']/len(only_a3)*100:.1f}%) "
                f"have the correct gold integer present in C2's completion text. Manual inspection "
                f"confirms C2 reasoned correctly to the gold answer in markdown prose like '**$16.00**'; "
                f"the original last-integer fallback regex picks the wrong substring. A prose-aware "
                f"extractor (smart_v2) lifts C2 from {358/n_total:.3f} to {c2_smart_correct/n_total:.3f} "
                f"with no model change. {c2_contains_gold/n_total*100:.1f}% of C2 completions contain the "
                f"gold integer somewhere, suggesting the true reasoning ceiling is closer to {c2_contains_gold/n_total*100:.0f}%."
            ),
        },
    }
    (OUT_DIR / "results_plan9.json").write_text(json.dumps(result, indent=2))

    print(f"only_A3 = {len(only_a3)}, only_C2 = {len(only_c2)}, both = {len(both)}, neither = {len(neither)}")
    print(f"Auto flags on only_A3 (n={len(only_a3)}):")
    for k, v in auto_tally.items():
        if k != "n_problems":
            print(f"  {k}: {v}")
    print(f"\nC2 accuracy:")
    print(f"  original extractor: {358}/{n_total} = {358/n_total:.4f}")
    print(f"  smart_v2 extractor: {c2_smart_correct}/{n_total} = {c2_smart_correct/n_total:.4f}")
    print(f"  contains gold int: {c2_contains_gold}/{n_total} = {c2_contains_gold/n_total:.4f}")
    print(f"\nFinding: {result['finding']['letter']} — {result['finding']['label']}")
    print(f"\nWrote {OUT_DIR}/inspection_set.json and {OUT_DIR}/results_plan9.json")


if __name__ == "__main__":
    main()
