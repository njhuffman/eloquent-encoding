"""Tests for dashboard metrics JSON upsert."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jepa.dashboard_metrics import upsert_stage_benchmarks


class TestUpsertStageBenchmarks(unittest.TestCase):
    def test_upsert_replaces_same_stage(self) -> None:
        d = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        p = d / "m_stage_benchmarks.json"
        upsert_stage_benchmarks(
            p,
            "m",
            {"sample_n": 100, "seed": 1},
            {"stage": 0, "top1_pct": 1.0, "n_positions": 10},
        )
        upsert_stage_benchmarks(
            p,
            "m",
            {"sample_n": 100, "seed": 1},
            {"stage": 1, "top1_pct": 5.0, "n_positions": 10},
        )
        upsert_stage_benchmarks(
            p,
            "m",
            {"sample_n": 200, "seed": 1},
            {"stage": 0, "top1_pct": 2.0, "n_positions": 20},
        )
        data = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(len(data["stages"]), 2)
        by_s = {int(x["stage"]): x for x in data["stages"]}
        self.assertEqual(by_s[0]["top1_pct"], 2.0)
        self.assertEqual(by_s[0]["n_positions"], 20)
        self.assertEqual(by_s[1]["top1_pct"], 5.0)
        self.assertEqual(data["sample_n"], 200)


if __name__ == "__main__":
    unittest.main()
