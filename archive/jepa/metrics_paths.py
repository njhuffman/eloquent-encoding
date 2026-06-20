"""Paths for per-epoch training metrics (JSONL) consumed by the dashboard."""

from __future__ import annotations

from pathlib import Path


def metrics_dir(checkpoint_dir: Path | str) -> Path:
    return Path(checkpoint_dir) / "metrics"


def epoch_metrics_jsonl_path(checkpoint_dir: Path | str, model_name: str, stage: int) -> Path:
    """One line per finished epoch: ``{checkpoint_dir}/metrics/{name}_stage_{stage}_epochs.jsonl``."""
    return metrics_dir(checkpoint_dir) / f"{model_name}_stage_{stage}_epochs.jsonl"


def model_profile_json_path(checkpoint_dir: Path | str, model_name: str) -> Path:
    """CPU single-forward + parameter count (written after stage 0)."""
    return metrics_dir(checkpoint_dir) / f"{model_name}_profile.json"


def stage_benchmarks_json_path(checkpoint_dir: Path | str, model_name: str) -> Path:
    """Per-stage move-ranking top-k metrics (val row + optional nested train sample from train_move_dataset_h5)."""
    return metrics_dir(checkpoint_dir) / f"{model_name}_stage_benchmarks.json"
