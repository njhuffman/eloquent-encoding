from __future__ import annotations

from pathlib import Path


def stage_checkpoint_path(checkpoint_dir: Path, model_name: str, stage: int) -> Path:
    return checkpoint_dir / f"{model_name}_stage_{stage}.pt"
