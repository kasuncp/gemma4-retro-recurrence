"""Phase 2 runner end-to-end smoke (CPU only).

We seed a fixture phase2 + phase1 results dir with synthetic JSONL,
then invoke ``path2_v2_phase2.py --summarize-only`` as a subprocess
and verify it produces a sensible summary + exit code + CSV + report
JSON.

Mirrors tests.path2_v2.test_phase1_runner. Three flavour fixtures:

  - All-preserved (3+) -> mode=normal, exit 0
  - Token-match fails  -> exit 1 (pre-flight halt)
  - All-broken         -> mode=halt, exit 1

Run:
    PYTHONPATH=. python -m unittest tests.path2_v2.test_phase2_runner -v
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "experiments" / "path2_v2_phase2.py"
NUM_LAYERS = 35


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

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


def _write_jsonl(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _seed_phase1_dir(d: Path):
    """Write the cached baseline-C2 JSONL Phase 2's pre-flight reads."""
    rows = [_gsm8k_row(i, completion=f"step {i}") for i in range(20)]
    _write_jsonl(d / "gsm8k__baseline-C2.jsonl", rows)


def _seed_phase2_dir(d: Path, *,
                      tokens_match: bool = True,
                      acc_per_layer: callable = None,
                      loop_per_layer: callable = None,
                      trunc_per_layer: callable = None):
    """Lay down 35 layer JSONLs + the pre-flight L17-r1 JSONL."""
    if acc_per_layer is None:
        acc_per_layer = lambda L: 0.72        # all preserved
    if loop_per_layer is None:
        loop_per_layer = lambda L: 0.0
    if trunc_per_layer is None:
        trunc_per_layer = lambda L: 0.0

    n = 50
    for L in range(NUM_LAYERS):
        acc = acc_per_layer(L)
        loop = loop_per_layer(L)
        trunc = trunc_per_layer(L)
        n_corr = int(round(acc * n))
        n_loop = int(round(loop * n))
        # truncation_rate counts cap-hit rows whose pred_smart_v2 is None.
        n_trunc = int(round(trunc * n))
        rows = []
        for i in range(n):
            is_correct = i < n_corr
            is_loop = i < n_loop
            # The trunc rows are AT THE END so they don't overlap the
            # correct ones, ensuring acc and trunc independently count.
            is_trunc = i >= (n - n_trunc)
            r = _gsm8k_row(i, correct_smart=is_correct, loop=is_loop,
                           truncated=is_trunc)
            if is_trunc:
                # truncated cap-hit with no extractable answer.
                r["pred_smart_v2"] = None
                r["correct_smart_v2"] = False
                r["n_gen_tokens"] = 512
            rows.append(r)
        _write_jsonl(d / f"gsm8k__L{L:02d}-r8.jsonl", rows)

    # Pre-flight L17-r1 cell: 20 rows. tokens_match controls whether
    # the completions equal phase1's baseline-C2.jsonl.
    rows = []
    for i in range(20):
        comp = f"step {i}" if tokens_match else f"DIFFERENT idx{i}"
        rows.append(_gsm8k_row(i, completion=comp))
    _write_jsonl(d / "gsm8k__L17-r1.jsonl", rows)


def _run_summarize(phase2_dir: Path, phase1_dir: Path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, str(RUNNER), "--summarize-only",
         "--output-dir", str(phase2_dir),
         "--phase1-output-dir", str(phase1_dir),
         "--no-plots"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, env=env,
        timeout=60,
    )


# ---------------------------------------------------------------------------
# Healthy: all preserved -> mode=normal, exit 0
# ---------------------------------------------------------------------------

class TestPhase2RunnerHealthy(unittest.TestCase):
    def test_all_preserved_passes(self):
        with tempfile.TemporaryDirectory() as base:
            base = Path(base)
            phase1 = base / "phase1"
            phase2 = base / "phase2"
            _seed_phase1_dir(phase1)
            _seed_phase2_dir(phase2, tokens_match=True)
            r = _run_summarize(phase2, phase1)
            self.assertEqual(r.returncode, 0,
                             f"runner failed unexpectedly:\n"
                             f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")

            # Summary JSON must exist and be parseable.
            sj = phase2 / "phase2_summary.json"
            self.assertTrue(sj.is_file())
            report = json.loads(sj.read_text())
            self.assertEqual(len(report["cells"]), NUM_LAYERS)
            self.assertEqual(report["phase3_launch"]["mode"], "normal")
            self.assertTrue(report["preflight"]["L17_r1_token_match"]["passed"])

            # CSV must exist and have 35 rows.
            csv_p = phase2 / "phase2_layer_map.csv"
            self.assertTrue(csv_p.is_file())
            text = csv_p.read_text().strip().splitlines()
            self.assertEqual(len(text), NUM_LAYERS + 1)  # +1 header

            # Every cell labelled "preserved".
            for name, c in report["cells"].items():
                self.assertEqual(c["bucket"], "preserved",
                                 f"{name} was {c['bucket']}, expected preserved")

    def test_console_table_present(self):
        with tempfile.TemporaryDirectory() as base:
            base = Path(base)
            phase1 = base / "phase1"; phase2 = base / "phase2"
            _seed_phase1_dir(phase1)
            _seed_phase2_dir(phase2)
            r = _run_summarize(phase2, phase1)
            self.assertIn("Phase 2 layer reasoning map", r.stdout)
            self.assertIn("Bucket counts", r.stdout)
            self.assertIn("Phase 3 launch decision", r.stdout)


# ---------------------------------------------------------------------------
# Pre-flight failure halts
# ---------------------------------------------------------------------------

class TestPhase2RunnerTokenMatchFails(unittest.TestCase):
    def test_token_mismatch_exits_1(self):
        with tempfile.TemporaryDirectory() as base:
            base = Path(base)
            phase1 = base / "phase1"; phase2 = base / "phase2"
            _seed_phase1_dir(phase1)
            _seed_phase2_dir(phase2, tokens_match=False)
            r = _run_summarize(phase2, phase1)
            self.assertEqual(r.returncode, 1,
                             f"runner should fail on token mismatch:\n{r.stdout}")
            sj = phase2 / "phase2_summary.json"
            report = json.loads(sj.read_text())
            self.assertFalse(report["preflight"]["L17_r1_token_match"]["passed"])


# ---------------------------------------------------------------------------
# All-broken layers -> halt
# ---------------------------------------------------------------------------

class TestPhase2RunnerHaltMode(unittest.TestCase):
    def test_all_broken_emits_halt(self):
        with tempfile.TemporaryDirectory() as base:
            base = Path(base)
            phase1 = base / "phase1"; phase2 = base / "phase2"
            _seed_phase1_dir(phase1)
            _seed_phase2_dir(
                phase2, tokens_match=True,
                acc_per_layer=lambda L: 0.10,
                loop_per_layer=lambda L: 0.80,
            )
            r = _run_summarize(phase2, phase1)
            self.assertEqual(r.returncode, 1)
            sj = phase2 / "phase2_summary.json"
            report = json.loads(sj.read_text())
            self.assertEqual(report["phase3_launch"]["mode"], "halt")
            self.assertEqual(len(report["buckets"]["preserved"]), 0)
            self.assertEqual(len(report["buckets"]["degraded"]), 0)
            self.assertEqual(len(report["buckets"]["broken"]), NUM_LAYERS)


# ---------------------------------------------------------------------------
# Mixed -> normal mode with subset of preserved
# ---------------------------------------------------------------------------

class TestPhase2RunnerMixed(unittest.TestCase):
    def test_three_preserved_picks_normal(self):
        with tempfile.TemporaryDirectory() as base:
            base = Path(base)
            phase1 = base / "phase1"; phase2 = base / "phase2"
            _seed_phase1_dir(phase1)

            preserved_layers = {15, 17, 19}
            degraded_layers = {8, 22}

            def acc(L):
                if L in preserved_layers:
                    return 0.70
                if L in degraded_layers:
                    return 0.50
                return 0.10

            def loop(L):
                # broken layers loop badly
                if L in preserved_layers or L in degraded_layers:
                    return 0.0
                return 0.80

            _seed_phase2_dir(phase2, tokens_match=True,
                             acc_per_layer=acc, loop_per_layer=loop)
            r = _run_summarize(phase2, phase1)
            self.assertEqual(r.returncode, 0)
            sj = phase2 / "phase2_summary.json"
            report = json.loads(sj.read_text())
            self.assertEqual(report["phase3_launch"]["mode"], "normal")
            self.assertEqual(set(report["buckets"]["preserved"]),
                             {f"L{L:02d}" for L in preserved_layers})
            self.assertEqual(set(report["buckets"]["degraded"]),
                             {f"L{L:02d}" for L in degraded_layers})


if __name__ == "__main__":
    unittest.main(verbosity=2)
