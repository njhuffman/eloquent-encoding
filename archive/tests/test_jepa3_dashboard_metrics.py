"""jepa3 dashboard benchmark JSON (joint top1 upsert)."""

from __future__ import annotations

import json
from pathlib import Path

import tempfile

from jepa3.dashboard_metrics import upsert_stage_benchmarks


def test_upsert_stage_benchmarks_joint_top1() -> None:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        p = d / "m_stage_benchmarks.json"
        upsert_stage_benchmarks(
            p,
            "m",
            {"sample_n": 100, "top1_pct_is_joint_from_to": True},
            {"stage": 0, "top1_pct": 1.0, "n_positions": 10, "benchmark": "joint_from_to_top1_packed"},
        )
        upsert_stage_benchmarks(
            p,
            "m",
            {"sample_n": 100},
            {"stage": 1, "top1_pct": 5.0, "n_positions": 10, "benchmark": "joint_from_to_top1_packed"},
        )
        upsert_stage_benchmarks(
            p,
            "m",
            {"sample_n": 200},
            {"stage": 0, "top1_pct": 2.0, "n_positions": 20, "benchmark": "joint_from_to_top1_packed"},
        )
        data = json.loads(p.read_text(encoding="utf-8"))
        by_s = {int(x["stage"]): x for x in data["stages"]}
        assert by_s[0]["top1_pct"] == 2.0
        assert by_s[0]["n_positions"] == 20
        assert by_s[1]["top1_pct"] == 5.0
        assert data["sample_n"] == 200
