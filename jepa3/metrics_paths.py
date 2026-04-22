"""Paths for jepa3 per-stage metrics (single JSON per training stage)."""

from __future__ import annotations

from pathlib import Path


def metrics_dir(checkpoint_dir: Path | str) -> Path:
    return Path(checkpoint_dir) / "metrics"


def stage_metrics_json_path(checkpoint_dir: Path | str, model_name: str, stage: int) -> Path:
    return metrics_dir(checkpoint_dir) / f"{model_name}_stage_{stage}_metrics.json"


def model_profile_json_path(checkpoint_dir: Path | str, model_name: str) -> Path:
    return metrics_dir(checkpoint_dir) / f"{model_name}_profile.json"


def stage_benchmarks_json_path(checkpoint_dir: Path | str, model_name: str) -> Path:
    return metrics_dir(checkpoint_dir) / f"{model_name}_stage_benchmarks.json"
