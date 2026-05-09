"""Phase 2 gate-evaluator tests --- CPU only.

The bucket boundary cases here ARE the contract Phase 2's pod run will
honor. If a band moves, the test must move and the plan must move.

Run:
    PYTHONPATH=. python -m unittest tests.path2_v2.test_phase2_gates -v
"""

import unittest

from probes import phase2_gates as g


def _gsm8k_summary(*, n=50, acc=0.72, acc_legacy=0.30, loop=0.0,
                   trunc=0.0, parse=1.0, mean_tok=120):
    """Mirror tests.path2_v2.test_phase1_gates._gsm8k_summary --- the
    summary contract is the same, the gate is what differs."""
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


class TestClassifyLayer(unittest.TestCase):
    def test_preserved_at_anchor(self):
        # Phase 1 1A measured 0.755; preserved threshold 0.65.
        s = _gsm8k_summary(acc=0.72, loop=0.00, trunc=0.00)
        bucket, reason = g.classify_layer(s)
        self.assertEqual(bucket, "preserved")
        self.assertIn("0.72", reason)

    def test_preserved_at_lower_boundary(self):
        s = _gsm8k_summary(acc=0.65, loop=0.00, trunc=0.00)
        bucket, _ = g.classify_layer(s)
        self.assertEqual(bucket, "preserved")

    def test_degraded_just_below_preserved(self):
        # Wilson +/- 13 pp puts 0.64 inside the noise band but the gate
        # is deterministic; document boundary acc=0.64 -> degraded.
        s = _gsm8k_summary(acc=0.64, loop=0.0, trunc=0.0)
        bucket, _ = g.classify_layer(s)
        self.assertEqual(bucket, "degraded")

    def test_degraded_at_lower_boundary(self):
        s = _gsm8k_summary(acc=0.40, loop=0.0, trunc=0.0)
        bucket, _ = g.classify_layer(s)
        self.assertEqual(bucket, "degraded")

    def test_broken_below_degraded(self):
        s = _gsm8k_summary(acc=0.25, loop=0.0, trunc=0.0)
        bucket, _ = g.classify_layer(s)
        self.assertEqual(bucket, "broken")

    def test_broken_when_loop_collapses(self):
        # Even with high accuracy, a loop_rate >= 0.50 is "structural
        # collapse" per the plan.
        s = _gsm8k_summary(acc=0.90, loop=0.50, trunc=0.0)
        bucket, _ = g.classify_layer(s)
        self.assertEqual(bucket, "broken")

    def test_broken_when_trunc_collapses(self):
        s = _gsm8k_summary(acc=0.90, loop=0.0, trunc=0.30)
        bucket, _ = g.classify_layer(s)
        self.assertEqual(bucket, "broken")

    def test_broken_with_just_below_loop_threshold_passes_pathology(self):
        # 0.49 < 0.50 -> NOT pathological for the loop test, but with
        # acc=0.20 the accuracy test still throws it into broken.
        s = _gsm8k_summary(acc=0.20, loop=0.49, trunc=0.0)
        bucket, _ = g.classify_layer(s)
        self.assertEqual(bucket, "broken")

    def test_no_rows_is_broken(self):
        bucket, reason = g.classify_layer({"n_problems": 0})
        self.assertEqual(bucket, "broken")
        self.assertEqual(reason, "no rows")

    def test_pathology_overrides_preserved_acc(self):
        # The whole point of the pathology gate: a model that emits
        # the right answer but is also looping is NOT preserved --- the
        # generation is unhealthy regardless of accuracy.
        s = _gsm8k_summary(acc=0.80, loop=0.60)
        bucket, _ = g.classify_layer(s)
        self.assertEqual(bucket, "broken")


class TestPhase3LaunchDecision(unittest.TestCase):
    def test_normal_when_three_preserved(self):
        b = {"preserved": ["L15", "L17", "L19"],
             "degraded": ["L08"], "broken": ["L00"]}
        d = g.decide_phase3_launch(b)
        self.assertEqual(d["mode"], "normal")
        self.assertEqual(d["anchors"], ["L15", "L17", "L19"])

    def test_narrow_when_two_preserved(self):
        b = {"preserved": ["L15", "L17"],
             "degraded": ["L08", "L09", "L10"], "broken": []}
        d = g.decide_phase3_launch(b)
        self.assertEqual(d["mode"], "narrow")
        self.assertEqual(d["anchors"], ["L15", "L17"])

    def test_narrow_when_one_preserved(self):
        b = {"preserved": ["L17"], "degraded": [], "broken": []}
        d = g.decide_phase3_launch(b)
        self.assertEqual(d["mode"], "narrow")

    def test_relaxed_when_zero_preserved_three_degraded(self):
        b = {"preserved": [], "degraded": ["L08", "L09", "L10"],
             "broken": ["L00", "L34"]}
        d = g.decide_phase3_launch(b)
        self.assertEqual(d["mode"], "relaxed")
        self.assertEqual(d["anchors"], ["L08", "L09", "L10"])

    def test_halt_when_zero_preserved_zero_degraded(self):
        b = {"preserved": [], "degraded": [], "broken": ["L00", "L34"]}
        d = g.decide_phase3_launch(b)
        self.assertEqual(d["mode"], "halt")
        self.assertEqual(d["anchors"], [])
        self.assertIn("retrofit", d["rationale"].lower())

    def test_halt_with_two_degraded(self):
        b = {"preserved": [], "degraded": ["L08", "L09"], "broken": []}
        d = g.decide_phase3_launch(b)
        # 0 preserved + 2 degraded < 3 -> halt
        self.assertEqual(d["mode"], "halt")


if __name__ == "__main__":
    unittest.main(verbosity=2)
