"""Resolve next training stage and validate model names for the jepa3 dashboard runner."""

from __future__ import annotations

from pathlib import Path

from jepa3.checkpoint_paths import stage_checkpoint_path
from jepa3.model_spec import MODEL_CONFIGS_DIR


def is_safe_model_name(name: str) -> bool:
    if not name or len(name) > 128:
        return False
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    return all(c in allowed for c in name)


def spec_file_stem_exists(name: str) -> bool:
    for ext in (".yaml", ".yml"):
        if (MODEL_CONFIGS_DIR / f"{name}{ext}").is_file():
            return True
    return False


def next_missing_stage(spec: dict) -> tuple[int | None, str]:
    name = spec["name"]
    ckpt_dir = Path(spec["checkpoint_dir"])
    n = len(spec["stages"])

    p0 = stage_checkpoint_path(ckpt_dir, name, 0)
    if not p0.is_file():
        return 0, "Next: init (stage 0)"

    for k in range(1, n + 1):
        pk = stage_checkpoint_path(ckpt_dir, name, k)
        pk_prev = stage_checkpoint_path(ckpt_dir, name, k - 1)
        if pk_prev.is_file() and not pk.is_file():
            return k, f"Next: train stage {k}"

    last = stage_checkpoint_path(ckpt_dir, name, n)
    if last.is_file():
        return None, "All stages complete"

    return None, "Inconsistent checkpoints (repair manually)"
