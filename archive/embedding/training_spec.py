"""
Load and validate MAE training JSON specs (all model/data/hyperparameters in one file).

CLI only supplies: which spec (--model / --config), then runtime overrides (device, workers, dirs).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from embedding.architectures import DEFAULT_ARCHITECTURE_ID
from embedding.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DATALOADER_NUM_WORKERS,
    LEARNING_RATE,
    LOG_INTERVAL,
    MAX_MASK_RATIO,
    MIN_MASK_RATIO,
    MODEL_CONFIGS_DIR,
    NUM_EPOCHS,
)

DEFAULT_TRAINING_SPEC: dict[str, Any] = {
    "masking": {
        "min_mask_ratio": MIN_MASK_RATIO,
        "max_mask_ratio": MAX_MASK_RATIO,
    },
    "architecture": {
        "id": DEFAULT_ARCHITECTURE_ID,
        "config": {},
    },
    "training": {
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "val_seed": 0,
        "in_memory": True,
        "log_interval": LOG_INTERVAL,
        "use_amp": True,
        "dataloader_num_workers": DATALOADER_NUM_WORKERS,
    },
    "outputs": {
        "checkpoint_dir": None,
        "register": False,
        "artifacts_dir": None,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def spec_path_for_model_name(name: str, repo_root: Path) -> Path:
    return (repo_root / MODEL_CONFIGS_DIR / f"{name}.json").resolve()


def load_raw_spec(path: Path) -> dict[str, Any]:
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e


def normalize_training_spec(
    raw: dict[str, Any],
    *,
    repo_root: Path,
    source_path: Path | None = None,
) -> dict[str, Any]:
    """
    Merge defaults, validate, resolve HDF5 paths to absolute strings.
    Returns a dict safe to JSON-serialize (paths as strings).
    """
    spec = _deep_merge(DEFAULT_TRAINING_SPEC, raw)
    name = spec.get("name")
    if not name or not isinstance(name, str):
        raise ValueError('Training spec must include a string "name" (model id).')
    name = name.strip()
    spec["name"] = name

    data = spec.get("data")
    if not isinstance(data, dict):
        raise ValueError('Training spec must include a "data" object with train_h5 and val_h5.')
    train_h5 = data.get("train_h5")
    val_h5 = data.get("val_h5")
    if not train_h5 or not val_h5:
        raise ValueError('data.train_h5 and data.val_h5 are required (paths to HDF5 splits).')

    def _resolve(p: str | Path) -> Path:
        path = Path(p)
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        return path.resolve()

    train_path = _resolve(train_h5)
    val_path = _resolve(val_h5)
    spec["data"] = {
        "train_h5": str(train_path),
        "val_h5": str(val_path),
    }

    arch = spec["architecture"]
    if not isinstance(arch.get("id"), str) or not arch["id"].strip():
        raise ValueError('architecture.id must be a non-empty string.')
    if not isinstance(arch.get("config"), dict):
        arch["config"] = {}

    m = spec["masking"]
    lo, hi = float(m["min_mask_ratio"]), float(m["max_mask_ratio"])
    if not (0.0 < lo <= hi < 1.0):
        raise ValueError("masking.min_mask_ratio / max_mask_ratio must satisfy 0 < min <= max < 1.")
    spec["masking"]["min_mask_ratio"] = lo
    spec["masking"]["max_mask_ratio"] = hi

    tr = spec["training"]
    for key in ("batch_size", "epochs", "val_seed", "log_interval", "dataloader_num_workers"):
        if key in tr:
            tr[key] = int(tr[key])
    tr["learning_rate"] = float(tr["learning_rate"])
    tr["in_memory"] = bool(tr.get("in_memory", True))
    tr["use_amp"] = bool(tr.get("use_amp", True))

    out = spec["outputs"]
    ckpt = out.get("checkpoint_dir")
    if ckpt:
        cp = Path(ckpt)
        if not cp.is_absolute():
            cp = (repo_root / cp).resolve()
        out["checkpoint_dir"] = str(cp)
    else:
        out["checkpoint_dir"] = str((repo_root / CHECKPOINT_DIR / name).resolve())

    art = out.get("artifacts_dir")
    if art:
        ap = Path(art)
        if not ap.is_absolute():
            ap = (repo_root / ap).resolve()
        out["artifacts_dir"] = str(ap)
    else:
        out["artifacts_dir"] = None

    out["register"] = bool(out.get("register", False))

    if source_path is not None:
        spec["spec_source_path"] = str(source_path.resolve())
    return spec


def load_training_spec(
    *,
    repo_root: Path,
    config_path: Path | None = None,
    model_name: str | None = None,
) -> tuple[dict[str, Any], Path]:
    """
    Load JSON training spec. Exactly one of config_path or model_name must be set.
    Returns (normalized_spec, path_to_source_file).
    """
    if (config_path is None) == (model_name is None):
        raise ValueError("Provide exactly one of config_path or model_name.")
    if config_path is not None:
        path = config_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Training spec not found: {path}")
    else:
        assert model_name is not None
        path = spec_path_for_model_name(model_name, repo_root)
        if not path.is_file():
            raise FileNotFoundError(
                f"No training spec for model {model_name!r} (expected {path}). "
                f"Create that file or pass --config path/to/spec.json."
            )
    raw = load_raw_spec(path)
    spec = normalize_training_spec(raw, repo_root=repo_root, source_path=path)
    if model_name is not None and spec["name"] != model_name:
        raise ValueError(
            f'Spec file {path} has "name": {spec["name"]!r} but --model was {model_name!r}; they must match.'
        )
    return spec, path


def apply_runtime_overrides(
    spec: dict[str, Any],
    *,
    workers: int | None = None,
    checkpoint_dir: Path | None = None,
    artifacts_dir: str | None = None,
    use_amp: bool | None = None,
    register: bool | None = None,
    repo_root: Path,
) -> dict[str, Any]:
    """Copy-on-write: apply CLI overrides to outputs.* and training.dataloader_num_workers / use_amp."""
    s = copy.deepcopy(spec)
    if workers is not None:
        s["training"]["dataloader_num_workers"] = int(workers)
    if checkpoint_dir is not None:
        s["outputs"]["checkpoint_dir"] = str(checkpoint_dir.resolve())
    if artifacts_dir is not None:
        ap = Path(artifacts_dir)
        if not ap.is_absolute():
            ap = (repo_root / ap).resolve()
        s["outputs"]["artifacts_dir"] = str(ap)
    if use_amp is not None:
        s["training"]["use_amp"] = bool(use_amp)
    if register is not None:
        s["outputs"]["register"] = bool(register)
    return s
