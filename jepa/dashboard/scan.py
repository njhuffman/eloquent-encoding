"""Discover model specs, checkpoint stage status, summaries, and epoch metrics JSONL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jepa.checkpoint_paths import stage_checkpoint_path
from jepa.metrics_paths import epoch_metrics_jsonl_path, model_profile_json_path, stage_benchmarks_json_path
from jepa.model_spec import MODEL_CONFIGS_DIR, load_model_spec


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def iter_model_spec_paths() -> list[Path]:
    paths: list[Path] = []
    if not MODEL_CONFIGS_DIR.is_dir():
        return paths
    for ext in (".yaml", ".yml"):
        paths.extend(sorted(MODEL_CONFIGS_DIR.glob(f"*{ext}")))
    return paths


def load_spec_by_model_name(name: str) -> dict[str, Any]:
    path = MODEL_CONFIGS_DIR / f"{name}.yaml"
    if not path.is_file():
        path = MODEL_CONFIGS_DIR / f"{name}.yml"
    if not path.is_file():
        raise FileNotFoundError(f"No spec for model {name!r}")
    return load_model_spec(path)


@dataclass
class StageStatus:
    stage: int
    checkpoint_path: str
    exists: bool
    blocked_by: int | None  # previous stage missing


def stage_grid_for_spec(spec: dict[str, Any]) -> list[StageStatus]:
    name = spec["name"]
    ckpt_dir = Path(spec["checkpoint_dir"])
    n_train = len(spec["stages"])
    # Stages 0..n_train inclusive (0 init, 1..n_train training)
    out: list[StageStatus] = []
    for s in range(0, n_train + 1):
        p = stage_checkpoint_path(ckpt_dir, name, s)
        blocked = None
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


def summarize_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        return {"error": "not_a_dict_checkpoint"}
    th = obj.get("train_hparams") or {}
    tm = obj.get("train_meta") or {}
    ts = obj.get("training_spec") or {}
    return {
        "train_hparams": {k: th[k] for k in th if k != "optimizer_state_dict"},
        "train_meta": {
            k: tm[k]
            for k in tm
            if k not in ("train_materialize_report", "val_materialize_report")
        },
        "training_spec": {
            "name": ts.get("name"),
            "stage": ts.get("stage"),
            "architecture": ts.get("architecture"),
        },
    }


def read_epoch_metrics_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def sparse_stage_points_from_checkpoints(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """One point per completed training stage from checkpoint train_hparams (no JSONL)."""
    name = spec["name"]
    ckpt_dir = Path(spec["checkpoint_dir"])
    points: list[dict[str, Any]] = []
    for s in range(1, len(spec["stages"]) + 1):
        p = stage_checkpoint_path(ckpt_dir, name, s)
        summ = summarize_checkpoint(p)
        if summ is None:
            continue
        th = summ.get("train_hparams") or {}
        if th.get("init_only"):
            continue
        points.append(
            {
                "stage": s,
                "epoch": th.get("best_epoch"),
                "val_loss": th.get("best_val_loss"),
                "train_loss": None,
                "sparse": True,
            }
        )
    return points


def read_model_profile(spec: dict[str, Any]) -> dict[str, Any] | None:
    path = model_profile_json_path(spec["checkpoint_dir"], spec["name"])
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def read_stage_benchmarks(spec: dict[str, Any]) -> dict[str, Any] | None:
    path = stage_benchmarks_json_path(spec["checkpoint_dir"], spec["name"])
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def list_models_catalog() -> list[dict[str, Any]]:
    paths = iter_model_spec_paths()
    out: list[dict[str, Any]] = []
    for p in paths:
        spec = load_model_spec(p)
        grid = stage_grid_for_spec(spec)
        profile = read_model_profile(spec)
        entry: dict[str, Any] = {
            "name": spec["name"],
            "spec_path": str(p),
            "architecture_id": spec["architecture"]["id"],
            "checkpoint_dir": spec["checkpoint_dir"],
            "n_training_stages": len(spec["stages"]),
            "stages": [
                {
                    "stage": s.stage,
                    "exists": s.exists,
                    "blocked_by": s.blocked_by,
                }
                for s in grid
            ],
        }
        if profile is not None:
            entry["n_parameters"] = profile.get("n_parameters")
            entry["cpu_single_forward_seconds"] = profile.get("cpu_single_forward_seconds")
        out.append(entry)
    return sorted(out, key=lambda x: x["name"])
