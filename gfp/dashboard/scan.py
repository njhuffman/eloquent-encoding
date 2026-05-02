"""Discover gfp model specs, checkpoint stages, and per-stage metrics JSON."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gfp.checkpoint_paths import stage_checkpoint_path
from gfp.metrics_paths import stage_metrics_json_path
from gfp.model_spec import MODEL_CONFIGS_DIR


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def is_safe_model_name(name: str) -> bool:
    if not name or len(name) > 128:
        return False
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    return all(c in allowed for c in name)


def iter_model_spec_paths() -> list[Path]:
    paths: list[Path] = []
    if not MODEL_CONFIGS_DIR.is_dir():
        return paths
    for ext in (".yaml", ".yml"):
        paths.extend(sorted(MODEL_CONFIGS_DIR.glob(f"*{ext}")))
    return paths


def list_model_names() -> list[str]:
    names: list[str] = []
    for p in iter_model_spec_paths():
        stem = p.stem
        if stem and stem not in names:
            names.append(stem)
    return sorted(names)


@dataclass
class StageStatus:
    stage: int
    checkpoint_path: str
    exists: bool
    blocked_by: int | None


def stage_grid_for_spec(spec: dict[str, Any]) -> list[StageStatus]:
    name = spec["name"]
    ckpt_dir = Path(spec["checkpoint_dir"])
    n_train = len(spec["stages"])
    out: list[StageStatus] = []
    for s in range(0, n_train + 1):
        p = stage_checkpoint_path(ckpt_dir, name, s)
        blocked: int | None = None
        if s > 0:
            prev = stage_checkpoint_path(ckpt_dir, name, s - 1)
            if not prev.is_file():
                blocked = s - 1
        out.append(
            StageStatus(
                stage=s,
                checkpoint_path=str(p),
                exists=p.is_file(),
                blocked_by=blocked if blocked is not None and not p.is_file() else None,
            )
        )
    return out


def read_stage_metrics_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def metrics_path_for_stage(spec: dict[str, Any], stage: int) -> Path:
    return stage_metrics_json_path(Path(spec["checkpoint_dir"]), spec["name"], stage)


def stage_status_as_dict(s: StageStatus) -> dict[str, Any]:
    return asdict(s)
