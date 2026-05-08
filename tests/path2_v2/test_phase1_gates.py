"""Phase 1 gate-evaluator tests --- CPU only.

Each gate is a pure function that takes a metrics dict and returns
``(passed, message)``. We test fixture metrics that should pass and
several that should fail, plus the aggregate ``all_passed`` helper.

The boundary cases here ARE the contract Phase 1's pod run will
honor; if you change a band, change the test, change the plan.

Run:
    PYTHONPATH=. python -m unittest tests.path2_v2.test_phase1_gates -v
"""

import unittest

from probes import phase1_gates as g


def _gsm8k_summary(*, n=50, acc=0.72, acc_legacy=0.30, loop=0.0,
                   trunc=0.0, parse=1.0, mean_tok=120):
    """Build a minimal GSM8K summary dict matching probes.eval_v3.summarise_gsm8k."""
    return {
        "n_problems": n,
        "accuracy_smart_v2": acc,
        "accuracy_legacy": acc_legacy,
        "extractor_lift": acc - acc_legacy,
        "truncation_rate": trunc,
        "loop_rate": loop,
        "parse_rate": parse,
        "mean_gen_tokens": mean_tok,
        "total_gen_seconds": 100.0,
        "mean_t_gen_seconds": 1.0,
        "pathology_flag": (trunc > 0.5) or (loop > 0.5),
    }


# ---------------------------------------------------------------------------
# 1A
# ---------------------------------------------------------------------------

class TestGate1A(unittest.TestCase):
    def test_passes_at_path1_anchor(self):
        s = _gsm8k_summary(acc=0.716, loop=0.00, trunc=0.00)
        ok, msg = g.gate_1A(s)
        self.assertTrue(ok, msg)

    def test_passes_at_band_lower(self):
        ok, _ = g.gate_1A(_gsm8k_summary(acc=0.65))
        self.assertTrue(ok)

    def test_passes_at_band_upper(self):
        ok, _ = g.gate_1A(_gsm8k_summary(acc=0.82))
        self.assertTrue(ok)

    def test_fails_below_band(self):
        ok, msg = g.gate_1A(_gsm8k_summary(acc=0.30))
        self.assertFalse(ok)
        self.assertIn("outside band", msg)

    def test_fails_high_loop_rate(self):
        ok, msg = g.gate_1A(_gsm8k_summary(loop=0.10))
        self.assertFalse(ok)
        self.assertIn("loop_rate", msg)

    def test_fails_high_truncation(self):
        ok, msg = g.gate_1A(_gsm8k_summary(trunc=0.10))
        self.assertFalse(ok)
        self.assertIn("truncation_rate", msg)

    def test_fails_low_parse(self):
        ok, msg = g.gate_1A(_gsm8k_summary(parse=0.50))
        self.assertFalse(ok)
        self.assertIn("parse_rate", msg)

    def test_fails_no_rows(self):
        ok, _ = g.gate_1A({"n_problems": 0})
        self.assertFalse(ok)


# ---------------------------------------------------------------------------
# 1B
# ---------------------------------------------------------------------------

class TestGate1B(unittest.TestCase):
    def test_passes_at_round5_anchor(self):
        s = _gsm8k_summary(acc=0.30, acc_legacy=0.548, loop=0.10, trunc=0.05)
        ok, msg = g.gate_1B(s)
        self.assertTrue(ok, msg)

    def test_fails_legacy_outside_band(self):
        s = _gsm8k_summary(acc=0.30, acc_legacy=0.10, loop=0.10)
        ok, msg = g.gate_1B(s)
        self.assertFalse(ok)
        self.assertIn("accuracy_legacy", msg)

    def test_fails_smart_outside_band(self):
        s = _gsm8k_summary(acc=0.05, acc_legacy=0.50, loop=0.10)
        ok, msg = g.gate_1B(s)
        self.assertFalse(ok)
        self.assertIn("accuracy_smart_v2", msg)

    def test_fails_loop_outside_band(self):
        s = _gsm8k_summary(acc=0.30, acc_legacy=0.50, loop=0.40)
        ok, msg = g.gate_1B(s)
        self.assertFalse(ok)
        self.assertIn("loop_rate", msg)


# ---------------------------------------------------------------------------
# 1C
# ---------------------------------------------------------------------------

class TestGate1C(unittest.TestCase):
    def test_full_match_passes(self):
        ok, msg = g.gate_1C({"shared": 20, "matches": 20})
        self.assertTrue(ok, msg)
        self.assertIn("matches=20/20", msg)

    def test_partial_match_fails(self):
        ok, msg = g.gate_1C({"shared": 20, "mismatches": [3, 7, 12]})
        self.assertFalse(ok)
        self.assertIn("3/20", msg)
        self.assertIn("[3, 7, 12]", msg)

    def test_no_shared_fails(self):
        ok, msg = g.gate_1C({"reason": "no shared idxs"})
        self.assertFalse(ok)
        self.assertIn("aborted", msg)

    def test_one_mismatch_fails_loudly(self):
        # The whole point of the gate: a single mismatch is FAIL,
        # not a tolerated jitter.
        ok, msg = g.gate_1C({"shared": 20, "mismatches": [5]})
        self.assertFalse(ok)


# ---------------------------------------------------------------------------
# 1D
# ---------------------------------------------------------------------------

class TestGate1D(unittest.TestCase):
    def test_healthy_smoke_passes(self):
        s = _gsm8k_summary(n=10, loop=0.30, mean_tok=400)
        ok, msg = g.gate_1D(s, expected_n=10)
        self.assertTrue(ok, msg)

    def test_partial_run_fails(self):
        s = _gsm8k_summary(n=7)
        ok, msg = g.gate_1D(s, expected_n=10)
        self.assertFalse(ok)
        self.assertIn("7/10", msg)

    def test_full_collapse_fails(self):
        s = _gsm8k_summary(n=10, loop=0.99)
        ok, msg = g.gate_1D(s, expected_n=10)
        self.assertFalse(ok)
        self.assertIn("collapse", msg)

    def test_immediate_eos_fails(self):
        s = _gsm8k_summary(n=10, mean_tok=0)
        ok, msg = g.gate_1D(s, expected_n=10)
        self.assertFalse(ok)
        self.assertIn("immediately", msg)

    def test_50_pct_loop_passes(self):
        # Phase 1 is generous on loop_rate; the gate only fires at 95 %+.
        # Phase 3 will tighten this.
        s = _gsm8k_summary(n=10, loop=0.50, mean_tok=300)
        ok, _ = g.gate_1D(s, expected_n=10)
        self.assertTrue(ok)


# ---------------------------------------------------------------------------
# 1E
# ---------------------------------------------------------------------------

class TestGate1E(unittest.TestCase):
    def test_zero_drift_passes(self):
        ok, _ = g.gate_1E(unmodified_ppl=12.5366, r1_ppl=12.5366)
        self.assertTrue(ok)

    def test_tiny_drift_passes(self):
        # 1e-6 relative drift, well under the 1e-4 cap
        ok, _ = g.gate_1E(unmodified_ppl=12.5366, r1_ppl=12.5366125)
        self.assertTrue(ok)

    def test_just_above_threshold_fails(self):
        # 1.5e-4 drift > 1e-4 cap. (Note: a1.001/10.0 - 1 lands at
        # ~9.99e-5 in IEEE double due to representation error, which
        # PASSES the gate; the threshold check is stable in the
        # well-clear cases we actually care about.)
        ok, _ = g.gate_1E(unmodified_ppl=10.0, r1_ppl=10.0015)
        self.assertFalse(ok)

    def test_big_drift_fails(self):
        ok, msg = g.gate_1E(unmodified_ppl=12.0, r1_ppl=15.0)
        self.assertFalse(ok)
        self.assertIn("drift", msg)

    def test_missing_inputs(self):
        ok, msg = g.gate_1E(None, 12.0)
        self.assertFalse(ok)
        ok, msg = g.gate_1E(12.0, None)
        self.assertFalse(ok)


# ---------------------------------------------------------------------------
# all_passed
# ---------------------------------------------------------------------------

class TestAllPassed(unittest.TestCase):
    def test_empty_report_fails(self):
        self.assertFalse(g.all_passed({"cells": {}}))

    def test_all_pass(self):
        report = {"cells": {
            "1A": {"gate_passed": True},
            "1B": {"gate_passed": True},
        }}
        self.assertTrue(g.all_passed(report))

    def test_one_fail(self):
        report = {"cells": {
            "1A": {"gate_passed": True},
            "1B": {"gate_passed": False},
        }}
        self.assertFalse(g.all_passed(report))


if __name__ == "__main__":
    unittest.main(verbosity=2)
