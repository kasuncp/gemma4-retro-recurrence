"""Phase 1 runner end-to-end smoke (CPU only).

We seed a fixture results dir with synthetic JSONL + the round-1 ppl
JSON, then invoke ``path2_v2_phase1.py --summarize-only`` as a
subprocess and verify it produces a sensible summary + exit code.

This is the closest we can get on a CPU laptop to "would Phase 1
pass on a healthy pod run". The numbers we seed are calibrated so
every gate passes, plus a "deliberately broken" variant where one
gate fails and the runner exits non-zero.

Run:
    PYTHONPATH=. python -m unittest tests.path2_v2.test_phase1_runner -v
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "experiments" / "path2_v2_phase1.py"


def _write_jsonl(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _gsm8k_row(idx, *, completion="answer is 16", n_tok=120,
               correct_smart=True, correct_legacy=True,
               truncated=False, loop=False):
    return {
        "idx": idx, "question": "Q?", "gold": "#### 16", "gold_int": 16,
        "completion": completion, "n_gen_tokens": n_tok,
        "pred_smart_v2": 16 if correct_smart else 0,
        "pred_legacy": 16 if correct_legacy else 0,
        "correct_smart_v2": correct_smart,
        "correct_legacy": correct_legacy,
        "truncated": truncated, "loop_flag": loop,
        "t_gen_seconds": 1.0,
    }


def _seed_phase1_dir(d: Path, *,
                      acc_1A=0.72, acc_legacy_1B=0.55, acc_smart_1B=0.30,
                      loop_1B=0.10,
                      tokens_match=True, loop_1D=0.30, drift_1E=1e-6):
    """Lay down a complete set of fixtures that should make every
    gate pass. Override individual fields to flip a single gate.
    """
    d = Path(d); d.mkdir(parents=True, exist_ok=True)

    # 1A: 50 GSM8K rows, accuracy = acc_1A
    n = 50
    n_correct = int(round(acc_1A * n))
    rows_1A = [_gsm8k_row(i, correct_smart=(i < n_correct)) for i in range(n)]
    _write_jsonl(d / "gsm8k__baseline-C2.jsonl", rows_1A)

    # 1B: 50 rows, with separate smart/legacy + a loop_rate
    n_legacy_correct = int(round(acc_legacy_1B * n))
    n_smart_correct = int(round(acc_smart_1B * n))
    n_loop = int(round(loop_1B * n))
    rows_1B = []
    for i in range(n):
        rows_1B.append(_gsm8k_row(
            i,
            correct_smart=(i < n_smart_correct),
            correct_legacy=(i < n_legacy_correct),
            loop=(i < n_loop),
        ))
    _write_jsonl(d / "gsm8k__baseline-8shot-control.jsonl", rows_1B)

    # 1C: W5-r1, 20 rows that match (or don't) the first 20 of 1A
    rows_1C = []
    for i in range(20):
        comp = rows_1A[i]["completion"] if tokens_match else f"DIFFERENT idx{i}"
        rows_1C.append(_gsm8k_row(i, completion=comp))
    _write_jsonl(d / "gsm8k__W5-r1.jsonl", rows_1C)

    # 1D: 10 rows of W5-r8 with adjustable loop rate
    n_d = 10
    n_loop_d = int(round(loop_1D * n_d))
    rows_1D = [
        _gsm8k_row(i, loop=(i < n_loop_d), n_tok=400)
        for i in range(n_d)
    ]
    _write_jsonl(d / "gsm8k__W5-r8.jsonl", rows_1D)

    # 1E: 1E_wikitext_smoke.json mimicking ple_sanity_check --mode original
    unmod = 12.5366
    r1_ppl = unmod * (1.0 + drift_1E)
    smoke = {
        "config": {"mode": "original", "model_id": "google/gemma-4-E2B"},
        "unmodified": {"ppl": unmod},
        "cells": [
            {"layer": 17, "r": 1, "ppl": r1_ppl}
        ],
    }
    (d / "1E_wikitext_smoke.json").write_text(json.dumps(smoke, indent=2))


def _run_summarize(d: Path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, str(RUNNER), "--summarize-only",
         "--output-dir", str(d)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, env=env,
        timeout=30,
    )


class TestPhase1RunnerHealthy(unittest.TestCase):
    def test_all_gates_pass(self):
        with tempfile.TemporaryDirectory() as d:
            _seed_phase1_dir(Path(d))
            r = _run_summarize(Path(d))
            self.assertEqual(r.returncode, 0,
                             f"runner failed unexpectedly:\n{r.stdout}\n{r.stderr}")
            self.assertIn("ALL GATES PASSED", r.stdout)
            self.assertIn("Proceed to Phase 2", r.stdout)

            # Summary JSON must exist and be parseable
            summary = json.loads((Path(d) / "phase1_summary.json").read_text())
            self.assertTrue(summary["all_gates_passed"])
            self.assertEqual(set(summary["cells"]),
                             {"1A_baseline_C2", "1B_baseline_8shot",
                              "1C_token_match", "1D_W5_r8_smoke",
                              "1E_base_ppl_smoke"})


class TestPhase1RunnerFailures(unittest.TestCase):
    def test_token_match_failure_halts(self):
        with tempfile.TemporaryDirectory() as d:
            _seed_phase1_dir(Path(d), tokens_match=False)
            r = _run_summarize(Path(d))
            self.assertEqual(r.returncode, 1)
            self.assertIn("HALT", r.stdout)
            self.assertIn("1C_token_match", r.stdout)

    def test_low_baseline_acc_halts(self):
        with tempfile.TemporaryDirectory() as d:
            _seed_phase1_dir(Path(d), acc_1A=0.30)
            r = _run_summarize(Path(d))
            self.assertEqual(r.returncode, 1)
            self.assertIn("1A_baseline_C2", r.stdout)

    def test_w5_r8_full_collapse_halts(self):
        with tempfile.TemporaryDirectory() as d:
            _seed_phase1_dir(Path(d), loop_1D=0.99)
            r = _run_summarize(Path(d))
            self.assertEqual(r.returncode, 1)
            self.assertIn("1D_W5_r8_smoke", r.stdout)

    def test_ppl_drift_halts(self):
        with tempfile.TemporaryDirectory() as d:
            _seed_phase1_dir(Path(d), drift_1E=1e-3)  # well over 1e-4
            r = _run_summarize(Path(d))
            self.assertEqual(r.returncode, 1)
            self.assertIn("1E_base_ppl_smoke", r.stdout)


class TestPhase1RunnerEmptyDir(unittest.TestCase):
    def test_no_jsonl_returns_failure(self):
        with tempfile.TemporaryDirectory() as d:
            r = _run_summarize(Path(d))
            self.assertEqual(r.returncode, 1)
            self.assertIn("HALT", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
