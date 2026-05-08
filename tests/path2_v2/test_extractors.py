"""Phase 0 extractor tests --- CPU only.

The 12 GSM8K cases below are the LOAD-BEARING contract for smart_v2.
They come from Path 1 plan 9's analysis of the 26 problems where the
legacy last-int extractor missed correct C2 answers. Any change to
smart_v2 must keep all 12 passing; if you need to relax one, justify
it on a Path-1-plan-9-equivalent fixture.

Run:
    PYTHONPATH=. python -m unittest tests.path2_v2.test_extractors -v
"""

import unittest

from probes.extractors import (
    extract_arc,
    flexible_extract_legacy,
    has_repetition_loop,
    is_truncated,
    score_arc_row,
    score_bbh_row,
    score_gsm8k_row,
    smart_v2,
)


class TestSmartV2(unittest.TestCase):
    """12 cases borrowed from plans/path_1_cot_tokens/plan9.md.

    Each case is (input_text, expected_extracted_int). ``None`` means
    "no integer extractable". These document what the extractor must
    do; not all of them are unique to smart_v2 vs legacy --- a few are
    "legacy already gets this", included so the test suite covers the
    common easy path too.
    """

    CASES = [
        # 1 — bold-with-dollar-and-decimal: the canonical recovery case
        ("The answer is **$16.00**.", 16),
        # 2 — last "= $N" anywhere; legacy also gets this
        ("So the customer saved $50 - $34 = $16.", 16),
        # 3 — bold integer alone
        ("**16**", 16),
        # 4 — "Final answer: N" pattern, no bold
        ("Final answer: 16", 16),
        # 5 — long CoT with the final bold-dollar number; legacy
        #     fails this one (would grab a 6 from "1.06") --- the
        #     headliner failure mode plan 9 fixed
        (
            "After 32% off, the price is $34. With tax: $34 × 1.06 = "
            "$36.04. The savings: $50 - $34 = **$16.00**.",
            16,
        ),
        # 6 — comma thousands
        ("The total cost is 1,250 dollars.", 1250),
        # 7 — negatives must round-trip
        ("Negative two: -2", -2),
        # 8 — multi-currency tokens, "total $N" hint
        ("$3.50 and $4.50, total $8.", 8),
        # 9 — "Step N: ...= K" stepped chain
        ("Step 1: 5+3=8. Step 2: 8*2=16.", 16),
        # 10 — empty
        ("", None),
        # 11 — no integers at all
        ("There is no number here.", None),
        # 12 — repetition loop case: extractor still returns SOMETHING;
        #      the loop_flag is the gate, not the extractor.
        ("Step 1: 5+3=8. " * 10, 8),
    ]

    def test_all_cases(self):
        for i, (text, expected) in enumerate(self.CASES, 1):
            with self.subTest(case=i, text=text[:40]):
                self.assertEqual(smart_v2(text), expected)


class TestFlexibleLegacy(unittest.TestCase):
    """Legacy extractor: last integer anywhere, with comma stripped.

    Used as a side-by-side reporting column to bridge to round 5
    numbers. Not the headline metric; we only check it returns a
    sensible last-int.
    """

    def test_basic(self):
        self.assertEqual(flexible_extract_legacy("5+3=8"), 8)

    def test_with_commas(self):
        self.assertEqual(flexible_extract_legacy("Total: 1,234 cookies."), 1234)

    def test_negative(self):
        self.assertEqual(flexible_extract_legacy("-7 then 3"), 3)

    def test_empty(self):
        self.assertIsNone(flexible_extract_legacy(""))

    def test_no_int(self):
        self.assertIsNone(flexible_extract_legacy("no numbers"))


class TestExtractArc(unittest.TestCase):
    CASES = [
        ("The answer is B.", "B"),
        ("(C)", "C"),
        ("After thinking, I'll pick option D.", "D"),
        ("blah blah\n\nAnswer: A", "A"),
        # Fall back to last standalone letter
        ("A is wrong, the right one is B.", "B"),
        # No letter at all
        ("No letter here", None),
    ]

    def test_all_cases(self):
        for text, expected in self.CASES:
            with self.subTest(text=text[:40]):
                self.assertEqual(extract_arc(text), expected)


class TestRepetitionLoop(unittest.TestCase):
    """Loop detector calibration --- Path 1 plan 3 canonical regex.

    Verified against the live Path 1 JSONL with this exact regex:
      - 8-shot CoT IT cells (A1-A4, N=500):  10.6% - 12.4%
      - C2 zero-shot IT cells (C1, C2):       0.0% (loop-free)
      - Path 2 round 5 recurrent (8-shot):    97-100%

    The 0% rate on C2 is part of why C2 beats 8-shot on this model
    --- the prompt format itself eliminates rep loops on IT. The
    Phase 1 baseline-C2 cell should similarly produce ~0%; any
    non-zero loop_rate on a recurrent config IS the recurrence
    contribution, isolated from prompt-induced loops.
    """

    CASES = [
        # (text, expected)
        ("normal completion no loop here", False),
        ("aaaaa bbbbb ccccc " * 5, True),     # 18-char block * 5
        ("Step 1: 5+3=8. " * 4, True),         # 14-char block * 4
        ("short", False),
        # NOTE: ("a" * 100) and ("ab" * 50) DO fire --- the regex
        # matches any 10-60 char window that repeats >=2 more times,
        # and "aaaaaaaaaa" (10 a's) repeats inside "a"*100. That
        # behaviour is correct; calibration anchor stays valid since
        # natural completions don't produce mono-char streaks of 30+
        # chars at the rates we care about.
        # Total length 9 of 'a' --- single char, can't form a 10-char
        # capturable substring, so no loop.
        ("a" * 9, False),
        # Edge: exactly 10 chars repeated 3 times = should fire
        ("0123456789" * 3, True),
        # Edge: 9 chars * 3 = should NOT fire (< 10 minimum)
        ("012345678" * 3, False),
    ]

    def test_all_cases(self):
        for text, expected in self.CASES:
            with self.subTest(text=text[:30]):
                self.assertEqual(has_repetition_loop(text), expected)


class TestTruncated(unittest.TestCase):
    def test_at_cap(self):
        self.assertTrue(is_truncated(512, max_new_tokens=512))

    def test_one_below(self):
        # 511 still counts (slack absorbs off-by-one in tokenisers)
        self.assertTrue(is_truncated(511, max_new_tokens=512))

    def test_two_below(self):
        # 510 is the conservative threshold path 1 used: still truncated
        self.assertTrue(is_truncated(510, max_new_tokens=512))

    def test_well_below(self):
        self.assertFalse(is_truncated(509, max_new_tokens=512))

    def test_legacy_256_cap(self):
        # Round 4's 256-cap regression
        self.assertTrue(is_truncated(255, max_new_tokens=256))
        self.assertFalse(is_truncated(253, max_new_tokens=256))


class TestScoreGsm8kRow(unittest.TestCase):
    def test_smart_recovers_legacy_misses(self):
        """The headline Path 1 plan 9 case: smart_v2 catches the bold
        $16.00. Legacy grabs the LAST integer in the string, which is
        the '00' from '$16.00' (= 0) --- a different wrong answer
        than the original plan-9 example, but still wrong, which is
        the point. smart_v2 lifts."""
        long_completion = (
            "After 32% off, the price is $34. With tax: $34 × 1.06 = "
            "$36.04. The savings: $50 - $34 = **$16.00**."
        )
        row = score_gsm8k_row(
            idx=42, question="?", gold="#### 16",
            completion=long_completion, n_gen_tokens=120,
        )
        self.assertEqual(row["pred_smart_v2"], 16)
        self.assertNotEqual(row["pred_legacy"], 16,
                            "legacy must miss this completion")
        self.assertTrue(row["correct_smart_v2"])
        self.assertFalse(row["correct_legacy"])

    def test_normal_row_agrees(self):
        completion = "Step: 8 * 2 = 16. So the answer is 16."
        row = score_gsm8k_row(
            idx=1, question="?", gold="#### 16",
            completion=completion, n_gen_tokens=20,
        )
        self.assertEqual(row["pred_smart_v2"], 16)
        self.assertEqual(row["pred_legacy"], 16)
        self.assertTrue(row["correct_smart_v2"])
        self.assertTrue(row["correct_legacy"])
        self.assertFalse(row["truncated"])
        self.assertFalse(row["loop_flag"])


class TestScoreArcRow(unittest.TestCase):
    def test_correct(self):
        row = score_arc_row(
            idx=0, question="?", choices=["x", "y", "z", "w"],
            gold="A", gold_letter="A",
            completion="The answer is A.", n_gen_tokens=10,
        )
        self.assertEqual(row["pred_letter"], "A")
        self.assertTrue(row["correct"])

    def test_wrong(self):
        row = score_arc_row(
            idx=0, question="?", choices=["x", "y", "z", "w"],
            gold="A", gold_letter="A",
            completion="(C) is correct.", n_gen_tokens=10,
        )
        self.assertEqual(row["pred_letter"], "C")
        self.assertFalse(row["correct"])


class TestScoreBbhRow(unittest.TestCase):
    def test_substring_match(self):
        row = score_bbh_row(
            idx="boolean_expressions/0", task="boolean_expressions",
            question="?", gold="True",
            completion="The expression evaluates to True.",
            n_gen_tokens=12,
        )
        self.assertTrue(row["correct"])

    def test_case_insensitive(self):
        row = score_bbh_row(
            idx="navigate/0", task="navigate",
            question="?", gold="Yes",
            completion="yes, return to start.",
            n_gen_tokens=12,
        )
        self.assertTrue(row["correct"])

    def test_no_match(self):
        row = score_bbh_row(
            idx="navigate/0", task="navigate",
            question="?", gold="Yes",
            completion="Negative.",
            n_gen_tokens=12,
        )
        self.assertFalse(row["correct"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
