"""Phase 2 heat-map plot + CSV writer tests.

Mostly verify file creation + CSV column ordering. The PNG render
test is skipped when matplotlib isn't installed --- the load-bearing
artefact for Phase 3 is the CSV.

Run:
    PYTHONPATH=. python -m unittest tests.path2_v2.test_layer_map_plot -v
"""

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

from experiments.path2_v2_layer_map_plot import CSV_FIELDS, write_csv


def _fake_report(num_layers: int = 35) -> dict:
    cells = {}
    buckets = {"preserved": [], "degraded": [], "broken": []}
    for L in range(num_layers):
        if L in (15, 17, 19):
            bucket = "preserved"; acc = 0.72; loop = 0.0; trunc = 0.0
        elif L in (8, 22):
            bucket = "degraded"; acc = 0.50; loop = 0.0; trunc = 0.0
        else:
            bucket = "broken"; acc = 0.10; loop = 0.80; trunc = 0.30
        name = f"L{L:02d}"
        buckets[bucket].append(name)
        cells[name] = {
            "summary": {
                "n_problems": 50,
                "accuracy_smart_v2": acc,
                "accuracy_smart_v2_ci95": [max(0.0, acc - 0.13),
                                           min(1.0, acc + 0.13)],
                "accuracy_legacy": acc - 0.05,
                "loop_rate": loop,
                "truncation_rate": trunc,
                "parse_rate": 1.0,
                "mean_gen_tokens": 240,
            },
            "metadata": {
                "attention_type": ("full_attention" if L % 5 == 0
                                   else "sliding_attention"),
                "is_kv_consumer": L >= 17,
                "depth_tertile": ("early" if L < 12 else
                                  "mid" if L < 24 else "late"),
            },
            "round2c_base_ppl": 12.5 + L * 0.1,
            "bucket": bucket,
            "bucket_reason": f"acc={acc:.2f} loop={loop:.2f} trunc={trunc:.2f}",
        }
    return {
        "cells": cells,
        "buckets": buckets,
        "phase3_launch": {
            "mode": "normal", "anchors": ["L15", "L17", "L19"],
            "rationale": "test fixture",
        },
        "preflight": {
            "L17_r1_token_match": {
                "shared": 20, "matches": 20, "passed": True,
            },
        },
    }


class TestWriteCsv(unittest.TestCase):
    def test_writes_one_row_per_layer(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "phase2_layer_map.csv"
            write_csv(report=_fake_report(35), path=out)
            self.assertTrue(out.is_file())
            with open(out) as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 35)

    def test_columns_match_contract(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "phase2_layer_map.csv"
            write_csv(report=_fake_report(5), path=out)
            with open(out) as fh:
                reader = csv.reader(fh)
                header = next(reader)
            self.assertEqual(header, CSV_FIELDS)

    def test_layers_sorted_ascending(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "phase2_layer_map.csv"
            write_csv(report=_fake_report(8), path=out)
            with open(out) as fh:
                rows = list(csv.DictReader(fh))
            layers = [int(r["layer"]) for r in rows]
            self.assertEqual(layers, list(range(8)))

    def test_bucket_passthrough(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "phase2_layer_map.csv"
            write_csv(report=_fake_report(35), path=out)
            with open(out) as fh:
                rows = {r["name"]: r for r in csv.DictReader(fh)}
            self.assertEqual(rows["L15"]["bucket"], "preserved")
            self.assertEqual(rows["L08"]["bucket"], "degraded")
            self.assertEqual(rows["L00"]["bucket"], "broken")


@unittest.skipIf(
    importlib.util.find_spec("matplotlib") is None,
    "matplotlib not installed; skipping PNG render test (CSV is "
    "the load-bearing artefact for Phase 3).",
)
class TestRenderPng(unittest.TestCase):
    def test_png_rendered_and_nonempty(self):
        from experiments.path2_v2_layer_map_plot import render_png
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "phase2_layer_map.png"
            render_png(report=_fake_report(35), path=out)
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 1024,
                               "PNG too small; likely empty figure")


if __name__ == "__main__":
    unittest.main(verbosity=2)
