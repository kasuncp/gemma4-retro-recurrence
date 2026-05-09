"""Phase 0 runtime smoke tests --- CPU only, no model load.

End-to-end sanity for the harness *minus* generation:
  - JSONL append-and-fsync, resume from existing rows
  - per-config summarisation, including pathology flags
  - the summarize-only entry point (path2_v2_eval.py --summarize-only)
    runs without importing torch.

Run:
    PYTHONPATH=. python -m unittest tests.path2_v2.test_runtime_smoke -v
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from probes import eval_v3
from probes.eval_v3 import (
    CONFIGS, append_row, cell_jsonl_path, cell_summary_path,
    existing_idxs, get_config, read_jsonl, summarise,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestConfigRegistry(unittest.TestCase):
    def test_baseline_c2_present(self):
        cfg = get_config("baseline-C2")
        self.assertEqual(cfg["prompt"], "C2")
        self.assertIsNone(cfg["block"])
        self.assertEqual(cfg["r"], 1)

    def test_w5_r1_is_loop_noop(self):
        cfg = get_config("W5-r1")
        self.assertEqual(cfg["block"], (15, 19))
        self.assertEqual(cfg["r"], 1)

    def test_w5_r8_block_intent(self):
        cfg = get_config("W5-r8")
        self.assertEqual(cfg["block"], (15, 19))
        self.assertEqual(cfg["r"], 8)

    def test_unknown_config_rejected(self):
        with self.assertRaises(ValueError):
            get_config("frobnicate")

    def test_get_config_returns_copy(self):
        cfg = get_config("baseline-C2")
        cfg["mutated"] = True
        cfg2 = get_config("baseline-C2")
        self.assertNotIn("mutated", cfg2)


class TestJsonlIO(unittest.TestCase):
    def test_append_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "gsm8k__baseline-C2.jsonl"
            append_row(p, {"idx": 0, "correct_smart_v2": True})
            append_row(p, {"idx": 1, "correct_smart_v2": False})
            self.assertEqual(existing_idxs(p), {0, 1})
            rows = read_jsonl(p)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["idx"], 0)

    def test_existing_idxs_empty(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "missing.jsonl"
            self.assertEqual(existing_idxs(p), set())


class TestSummariseGsm8k(unittest.TestCase):
    def _row(self, idx, smart_correct, legacy_correct, n_tok=120,
             trunc=False, loop=False, parsed=True):
        return {
            "idx": idx, "n_gen_tokens": n_tok,
            "pred_smart_v2": (1 if parsed else None),
            "pred_legacy": (1 if legacy_correct or smart_correct else None),
            "correct_smart_v2": smart_correct,
            "correct_legacy": legacy_correct,
            "truncated": trunc, "loop_flag": loop,
            "t_gen_seconds": 1.0,
        }

    def test_basic(self):
        rows = [
            self._row(0, True,  True),
            self._row(1, True,  False),  # smart_v2 lift
            self._row(2, False, False),
        ]
        s = summarise(rows, "gsm8k")
        self.assertEqual(s["n_problems"], 3)
        self.assertAlmostEqual(s["accuracy_smart_v2"], 2/3)
        self.assertAlmostEqual(s["accuracy_legacy"], 1/3)
        self.assertAlmostEqual(s["extractor_lift"], 1/3)

    def test_pathology_flag_truncation(self):
        # Pathology = "harness couldn't surface an answer." The 8 trunc
        # rows must also have parsed=False; otherwise they're verbose-
        # but-extracted rows (accuracy concern, not pathology).
        rows = [self._row(i, False, False, trunc=True, parsed=False)
                for i in range(8)]
        rows += [self._row(i, True, True, trunc=False) for i in range(2)]
        s = summarise(rows, "gsm8k")
        self.assertTrue(s["pathology_flag"])
        self.assertAlmostEqual(s["truncation_rate"], 0.8)

    def test_truncated_but_parsed_does_not_count(self):
        # The Phase 1 disambiguation: a row that hit the cap but still
        # produced a parseable answer is NOT a truncation pathology.
        # All 10 rows hit the cap, all 10 have an extractable answer
        # (correct or wrong) -> truncation_rate must be 0.
        rows = [self._row(i, smart_correct=False, legacy_correct=False,
                          trunc=True, parsed=True)
                for i in range(10)]
        s = summarise(rows, "gsm8k")
        self.assertAlmostEqual(s["truncation_rate"], 0.0)
        self.assertFalse(s["pathology_flag"])

    def test_pathology_flag_loop(self):
        rows = [self._row(i, True, True, loop=True) for i in range(6)]
        rows += [self._row(i, True, True, loop=False) for i in range(4)]
        s = summarise(rows, "gsm8k")
        self.assertTrue(s["pathology_flag"])

    def test_no_pathology(self):
        rows = [self._row(i, True, True) for i in range(10)]
        s = summarise(rows, "gsm8k")
        self.assertFalse(s["pathology_flag"])

    def test_empty(self):
        s = summarise([], "gsm8k")
        self.assertEqual(s, {"n_problems": 0})


class TestSummariseArcAndBbh(unittest.TestCase):
    def test_arc(self):
        rows = [
            {"idx": 0, "correct": True,  "pred_letter": "A",
             "n_gen_tokens": 5, "truncated": False, "loop_flag": False,
             "t_gen_seconds": 0.1},
            {"idx": 1, "correct": False, "pred_letter": "C",
             "n_gen_tokens": 5, "truncated": False, "loop_flag": False,
             "t_gen_seconds": 0.1},
            {"idx": 2, "correct": True,  "pred_letter": None,
             "n_gen_tokens": 5, "truncated": True, "loop_flag": False,
             "t_gen_seconds": 0.1},
        ]
        s = summarise(rows, "arc-c")
        self.assertEqual(s["n_problems"], 3)
        self.assertAlmostEqual(s["accuracy"], 2/3)
        self.assertAlmostEqual(s["parse_rate"], 2/3)
        self.assertAlmostEqual(s["truncation_rate"], 1/3)

    def test_bbh_per_task(self):
        rows = [
            {"idx": "object_counting/0", "task": "object_counting",
             "correct": True, "n_gen_tokens": 5,
             "truncated": False, "loop_flag": False, "t_gen_seconds": 0.1},
            {"idx": "object_counting/1", "task": "object_counting",
             "correct": False, "n_gen_tokens": 5,
             "truncated": False, "loop_flag": False, "t_gen_seconds": 0.1},
            {"idx": "navigate/0", "task": "navigate",
             "correct": True, "n_gen_tokens": 5,
             "truncated": False, "loop_flag": False, "t_gen_seconds": 0.1},
        ]
        s = summarise(rows, "bbh-lite")
        self.assertAlmostEqual(s["accuracy"], 2/3)
        self.assertAlmostEqual(s["per_task_accuracy"]["object_counting"], 0.5)
        self.assertAlmostEqual(s["per_task_accuracy"]["navigate"], 1.0)


class TestSummarizeOnlyEntryPoint(unittest.TestCase):
    """The CPU-laptop path 0 acceptance gate: --summarize-only must
    work without torch installed. We test by running the script as a
    subprocess and asserting it exits 0 and prints the table.
    """

    def test_cli_summarize_only(self):
        with tempfile.TemporaryDirectory() as d:
            # Seed a fixture cell with two rows
            jsonl = Path(d) / "gsm8k__baseline-C2.jsonl"
            append_row(jsonl, {
                "idx": 0, "n_gen_tokens": 100,
                "pred_smart_v2": 16, "pred_legacy": 16,
                "correct_smart_v2": True, "correct_legacy": True,
                "truncated": False, "loop_flag": False,
                "t_gen_seconds": 1.0,
            })
            append_row(jsonl, {
                "idx": 1, "n_gen_tokens": 200,
                "pred_smart_v2": None, "pred_legacy": None,
                "correct_smart_v2": False, "correct_legacy": False,
                "truncated": False, "loop_flag": True,
                "t_gen_seconds": 2.0,
            })
            # Subprocess so we genuinely test the entry point's
            # imports don't reach into torch on this code path.
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_ROOT)
            r = subprocess.run(
                [sys.executable, "experiments/path2_v2_eval.py",
                 "--summarize-only", "--output-dir", d],
                cwd=str(REPO_ROOT),
                capture_output=True, text=True, env=env,
                timeout=30,
            )
            self.assertEqual(
                r.returncode, 0,
                f"non-zero exit: stdout={r.stdout!r} stderr={r.stderr!r}",
            )
            self.assertIn("Path 2 v2 summary", r.stdout)
            self.assertIn("baseline-C2", r.stdout)


class TestCellPaths(unittest.TestCase):
    def test_jsonl_path(self):
        p = cell_jsonl_path(
            output_dir=Path("/tmp/x"),
            benchmark="gsm8k", config_name="baseline-C2",
        )
        self.assertEqual(p.name, "gsm8k__baseline-C2.jsonl")

    def test_summary_path(self):
        p = cell_summary_path(
            output_dir=Path("/tmp/x"),
            benchmark="arc-c", config_name="W5-r8",
        )
        self.assertEqual(p.name, "arc-c__W5-r8.summary.json")


if __name__ == "__main__":
    unittest.main(verbosity=2)
