"""Paths for rfp per-stage metrics JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def metrics_dir(checkpoint_dir: Path | str) -> Path:
    return Path(checkpoint_dir) / "metrics"


def stage_metrics_json_path(checkpoint_dir: Path | str, model_name: str, stage: int) -> Path:
    return metrics_dir(checkpoint_dir) / f"{model_name}_stage_{stage}_metrics.json"


def write_stage_metrics_json(path: Path, record: dict[str, Any]) -> Path:
    text = json.dumps(record, indent=2, default=str)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(text, encoding="utf-8")
        return path
    except PermissionError:
        name = record.get("model")
        if not isinstance(name, str) or not name.strip():
            raise
        repo_root = Path(__file__).resolve().parents[1]
        alt = repo_root / "rfp_checkpoints" / name.strip() / "metrics" / path.name
        alt.parent.mkdir(parents=True, exist_ok=True)
        alt.write_text(text, encoding="utf-8")
        print(
            f"Warning: could not write metrics to {path}; wrote to {alt} (fix ownership or checkpoint_dir).",
            file=sys.stderr,
        )
        return alt
