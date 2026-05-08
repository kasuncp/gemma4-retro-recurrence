"""Path-1-validated extractors and pathology detectors for path 2 v2.

Borrowed verbatim from
  - experiments/path1_c2_vs_a3_inspection.py  (smart_v2, _BOLD_RE,
    _ANSWER_RE, _EQ_END_RE)  --- N=500 GSM8K validated, 0.716 -> 0.780
  - experiments/path1_rep_loop_analysis.py    (CANONICAL_REP_RE)
    --- baseline 9-10% on the IT model, 97-100% on round 5 recurrent
    configs.

Do not modify regexes here without re-running those test suites.
The 12 GSM8K cases in tests/path2_v2/test_extractors.py are the
contract: any change must keep them all passing.
"""

import re
from typing import Optional


# ---------------------------------------------------------------------------
# GSM8K integer extractors
# ---------------------------------------------------------------------------

_BOLD_RE = re.compile(r"\*\*[^*]*?\$?(-?\d+)(?:\.\d+)?[^*]*?\*\*")
_ANSWER_RE = re.compile(
    r"(?:[Aa]nswer|[Tt]otal|[Ff]inal).{0,80}?\$?(-?\d+)(?:\.\d+)?"
)
_EQ_END_RE = re.compile(
    r"=\s*\$?(-?\d+)(?:\.\d+)?\s*(?:[A-Za-z%]+)?\s*\.?\s*$",
    re.MULTILINE,
)
_INT_RE = re.compile(r"-?\d+")


def smart_v2(text: str) -> Optional[int]:
    """Path 1 plan 9 extractor.

    Apply rules in order, first match wins. The order matters --- it
    encodes the empirical finding that the LAST bold number is more
    often the model's intended answer than any prior number, even if
    the model wrote "= 16" earlier in the chain.

    Returns ``None`` for empty or no-integer-anywhere strings; an
    ``int`` otherwise. Decimal numbers are truncated to int (the
    extractor was tuned for GSM8K, where every answer is integer or
    integer-valued dollar amounts).
    """
    if not text:
        return None
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

    nums = _INT_RE.findall(norm)
    return int(nums[-1]) if nums else None


def flexible_extract_legacy(text: str) -> Optional[int]:
    """Last-integer fallback used by round 4/5.

    Kept so every cell can report a "round-5-style accuracy" column
    side-by-side with smart_v2 for cross-round bridging without
    re-running anything.
    """
    if not text:
        return None
    nums = _INT_RE.findall(text.replace(",", ""))
    return int(nums[-1]) if nums else None


def gold_int(gold) -> Optional[int]:
    """Gold-side extractor for GSM8K: the dataset stores answers as
    a free-form string ending in '#### N'; older snapshots store an
    int directly. Return an int or None."""
    if isinstance(gold, (int, float)):
        return int(gold)
    if not isinstance(gold, str):
        return None
    g = gold.replace(",", "")
    m = re.search(r"####\s*(-?\d+)", g)
    if m:
        return int(m.group(1))
    nums = _INT_RE.findall(g)
    return int(nums[-1]) if nums else None


# ---------------------------------------------------------------------------
# ARC-Challenge / ARC-Easy letter extractor
# ---------------------------------------------------------------------------

_ARC_PRIMARY = re.compile(
    r"(?:answer\s*[:\-]?\s*|\bthe answer is\s+|\boption\s+|\bchoice\s+)\(?([A-D])\)?",
    re.IGNORECASE,
)
_ARC_FALLBACK = re.compile(r"\b([A-D])\b")


def extract_arc(text: str) -> Optional[str]:
    """Return the predicted ARC letter (A-D) or None.

    Strategy:
      1. Look for an explicit "answer/option/choice <letter>" cue
         in the last 200 chars (greedy: the model usually states
         the answer at the end of its completion).
      2. Fall back to the LAST standalone A-D letter anywhere.
      3. Return None if neither matches.
    """
    if not text:
        return None
    tail = text[-200:]
    m = _ARC_PRIMARY.search(tail)
    if m:
        return m.group(1).upper()
    last = _ARC_FALLBACK.findall(text)
    return last[-1].upper() if last else None


# ---------------------------------------------------------------------------
# Pathology detectors
# ---------------------------------------------------------------------------

# Path 1 plan 3 / rep_loop_analysis canonical regex.
# 10-60 char block repeated 3+ times.
_LOOP_RE = re.compile(r"(.{10,60})\1{2,}", re.DOTALL)


def has_repetition_loop(text: Optional[str]) -> bool:
    """True iff a 10-60 char substring repeats at least 3 times.

    Calibration anchors (verified against Path 1 cells with this
    exact regex + DOTALL):
      - Path 1 8-shot CoT IT (A1-A4 cells, N=500):  10.6 - 12.4%
      - Path 1 C2 zero-shot IT (C1, C2 cells):       0.0% --- C2
        eliminates rep loops on this model entirely. This is part
        of *why* C2 beats 8-shot CoT (71.6% vs 30.0% on GSM8K).
      - Path 2 round 5 recurrent configs (8-shot, r=8):  97-100%
        --- but confounded by the broken prompt; it's an open
        question how much loop-rate the recurrent block contributes
        on top of a healthy C2 baseline.

    Use as a per-completion flag; aggregate to ``loop_rate`` per
    config. A config with ``loop_rate > 0.30`` should be reported
    as "generation pathology" before its accuracy is interpreted.
    Path 2 v2 Phase 1 expects baseline-C2 ~0% and W5-r8 unknown;
    that delta IS the recurrence-induced loop signal.
    """
    return bool(_LOOP_RE.search(text or ""))


def is_truncated(n_gen_tokens: int, max_new_tokens: int = 512) -> bool:
    """True iff generation hit (or effectively hit) the max-new-tokens cap.

    The ``- 2`` slack absorbs off-by-one in tokenizer post-processing
    (some pipelines append a sentinel that doesn't count toward
    ``max_new_tokens`` but bumps ``n_gen_tokens`` by 1). Path 1 plan 9
    used ``>= 510`` against a 512 cap; this is the same threshold.

    Calibration: round 4 with ``max_new_tokens=256`` truncated 155/238
    wrong GSM8K baselines; the 512 cap matters because every generation
    has to either answer or self-terminate within it.
    """
    return n_gen_tokens >= max(1, max_new_tokens - 2)


# ---------------------------------------------------------------------------
# Per-row scoring helper (used by the eval loop and the smoke test alike)
# ---------------------------------------------------------------------------

def score_gsm8k_row(
    *, idx, question, gold, completion: str, n_gen_tokens: int,
    max_new_tokens: int = 512,
) -> dict:
    """Build the canonical per-problem dict for a GSM8K cell.

    Pure function --- no I/O, no model. Tested against fixtures.
    """
    pred_smart = smart_v2(completion)
    pred_legacy = flexible_extract_legacy(completion)
    g = gold_int(gold)
    return {
        "idx": idx,
        "question": question,
        "gold": gold,
        "gold_int": g,
        "completion": completion,
        "n_gen_tokens": n_gen_tokens,
        "pred_smart_v2": pred_smart,
        "pred_legacy": pred_legacy,
        "correct_smart_v2": (pred_smart is not None and g is not None and pred_smart == g),
        "correct_legacy":   (pred_legacy is not None and g is not None and pred_legacy == g),
        "truncated": is_truncated(n_gen_tokens, max_new_tokens),
        "loop_flag": has_repetition_loop(completion),
    }


def score_arc_row(
    *, idx, question, choices, gold, gold_letter, completion: str,
    n_gen_tokens: int, max_new_tokens: int = 512,
) -> dict:
    pred = extract_arc(completion)
    return {
        "idx": idx,
        "question": question,
        "choices": choices,
        "gold": gold,
        "gold_letter": gold_letter,
        "completion": completion,
        "n_gen_tokens": n_gen_tokens,
        "pred_letter": pred,
        "correct": pred is not None and gold_letter is not None and pred == gold_letter,
        "truncated": is_truncated(n_gen_tokens, max_new_tokens),
        "loop_flag": has_repetition_loop(completion),
    }


def score_bbh_row(
    *, idx, task, question, gold, completion: str, n_gen_tokens: int,
    max_new_tokens: int = 512,
) -> dict:
    """BBH scoring is task-dependent in general (some tasks are MC,
    some are spans). For the 5 lite tasks Phase 3 uses, the gold
    answers are short strings the model has to reproduce; we score
    by case-insensitive substring match in the last 100 chars of
    the completion. This matches Path 1 plan 7's approach.
    """
    g = (gold or "").strip()
    tail = (completion or "")[-200:].strip()
    correct = bool(g) and g.lower() in tail.lower()
    return {
        "idx": idx,
        "task": task,
        "question": question,
        "gold": gold,
        "completion": completion,
        "n_gen_tokens": n_gen_tokens,
        "correct": correct,
        "truncated": is_truncated(n_gen_tokens, max_new_tokens),
        "loop_flag": has_repetition_loop(completion),
    }
