"""Phase 1 token-match diff tool tests --- CPU only.

Covers the three exit codes:
  0 -- all shared idxs byte-equal
  1 -- one or more mismatches
  2 -- no shared idxs (path or schema error)

The tool is invoked as a subprocess so we exercise the actual CLI
contract Phase 1's runner depends on.

Run:
    PYTHONPATH=. python -m unittest tests.path2_v2.test_token_match -v
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL = REPO_ROOT / "experiments" / "path2_v2_token_match.py"


def _write_jsonl(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _run(argv):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        argv, cwd=str(REPO_ROOT), capture_output=True, text=True, env=env,
        timeout=20,
    )


class TestTokenMatch(unittest.TestCase):
    def test_match_all(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rows = [{"idx": i, "completion": f"answer is {i}"} for i in range(5)]
            _write_jsonl(d / "left.jsonl", rows)
            _write_jsonl(d / "right.jsonl", rows)
            r = _run([sys.executable, str(TOOL),
                      "--left", str(d / "left.jsonl"),
                      "--right", str(d / "right.jsonl"),
                      "--limit", "5"])
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("MATCH 5/5", r.stdout)

    def test_one_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            left = [{"idx": i, "completion": f"answer is {i}"} for i in range(5)]
            right = list(left)
            right[2] = {"idx": 2, "completion": "different completion"}
            _write_jsonl(d / "left.jsonl", left)
            _write_jsonl(d / "right.jsonl", right)
            r = _run([sys.executable, str(TOOL),
                      "--left", str(d / "left.jsonl"),
                      "--right", str(d / "right.jsonl"),
                      "--limit", "5"])
            self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
            self.assertIn("FAIL", r.stdout)
            self.assertIn("idx=2", r.stdout)

    def test_no_shared_idxs(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _write_jsonl(d / "left.jsonl",
                         [{"idx": i, "completion": "x"} for i in range(3)])
            _write_jsonl(d / "right.jsonl",
                         [{"idx": i + 100, "completion": "y"} for i in range(3)])
            r = _run([sys.executable, str(TOOL),
                      "--left", str(d / "left.jsonl"),
                      "--right", str(d / "right.jsonl")])
            self.assertEqual(r.returncode, 2)
            self.assertIn("FAIL", r.stdout)

    def test_missing_left(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _write_jsonl(d / "right.jsonl", [{"idx": 0, "completion": "x"}])
            r = _run([sys.executable, str(TOOL),
                      "--left", str(d / "missing.jsonl"),
                      "--right", str(d / "right.jsonl")])
            self.assertEqual(r.returncode, 2)

    def test_limit_truncates_comparison(self):
        """If --limit < total shared idxs, only the first N are compared.

        Mismatches outside the limit must NOT cause failure.
        """
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            left = [{"idx": i, "completion": "ok"} for i in range(10)]
            right = [{"idx": i, "completion": "ok"} for i in range(10)]
            right[7]["completion"] = "mismatch beyond limit"
            _write_jsonl(d / "left.jsonl", left)
            _write_jsonl(d / "right.jsonl", right)
            r = _run([sys.executable, str(TOOL),
                      "--left", str(d / "left.jsonl"),
                      "--right", str(d / "right.jsonl"),
                      "--limit", "5"])
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("MATCH 5/5", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
