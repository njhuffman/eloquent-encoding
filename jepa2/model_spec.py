"""YAML model spec for jepa2: defaults + per-stage deep merge (no materialized negatives)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIGS_DIR = _REPO_ROOT / "jepa2" / "model_configs"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


DEFAULTS: dict[str, Any] = {
    "ema_momentum": 0.999,
    "M_train": 64,
    "M_eval": 64,
    "ce_weight": 1.0,
    "mse_played_weight": 0.1,
    "ce_label_smoothing": 0.0,
    "batch_size": 256,
    "weight_decay": 0.05,
    "dataloader_num_workers": 0,
    "log_interval": 100,
    "use_amp": True,
    "vicreg": {
        "inv_coef": 0.0,
        "var_coef": 0.1,
        "cov_coef": 0.0,
        "std_target": 1.0,
    },
    "val_legal_seed": 42,
}


def load_model_spec(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("model spec root must be a mapping")
    name = raw.get("name")
    if not name or not isinstance(name, str):
        raise ValueError('spec must have string "name" (same as --model)')
    raw["name"] = name.strip()

    if "architecture" not in raw or not isinstance(raw["architecture"], dict):
        raise ValueError('spec must have "architecture" object')
    if not raw["architecture"].get("id"):
        raise ValueError("architecture.id required")

    def _resolve_h5_key(key: str) -> str:
        p = Path(raw.get(key, ""))
        if not p.parts:
            raise ValueError(f"{key} required (path to move-sample HDF5)")
        return str(p.expanduser().resolve() if p.is_absolute() else (_REPO_ROOT / p).resolve())

    raw["train_move_dataset_h5"] = _resolve_h5_key("train_move_dataset_h5")
    raw["val_move_dataset_h5"] = _resolve_h5_key("val_move_dataset_h5")

    vs = raw.get("val_sample")
    if not isinstance(vs, dict) or "n" not in vs or "seed" not in vs:
        raise ValueError('val_sample: {n, seed} required')
    vs["n"] = int(vs["n"])
    vs["seed"] = int(vs["seed"])
    raw["val_sample"] = vs

    ckpt = raw.get("checkpoint_dir")
    if not ckpt:
        raw["checkpoint_dir"] = str((_REPO_ROOT / "jepa2_checkpoints" / name).resolve())
    else:
        p = Path(ckpt)
        raw["checkpoint_dir"] = str(p.expanduser().resolve() if p.is_absolute() else (_REPO_ROOT / p).resolve())

    merged_defaults = _deep_merge(copy.deepcopy(DEFAULTS), raw.get("defaults") or {})
    raw["defaults"] = merged_defaults

    dm = _deep_merge(
        {
            "move_benchmark_sample_n": 2048,
            "move_benchmark_seed": 42,
            "move_benchmark_train_seed": 1000045,
            "move_benchmark_succ_chunk": 256,
            "device": "auto",
        },
        raw.get("dashboard_metrics") or {},
    )
    dm["move_benchmark_sample_n"] = int(dm["move_benchmark_sample_n"])
    dm["move_benchmark_seed"] = int(dm["move_benchmark_seed"])
    dm["move_benchmark_train_seed"] = int(dm["move_benchmark_train_seed"])
    dm["move_benchmark_succ_chunk"] = int(dm["move_benchmark_succ_chunk"])
    if dm["device"] not in ("auto", "cuda", "cpu"):
        raise ValueError('dashboard_metrics.device must be "auto", "cuda", or "cpu"')
    raw["dashboard_metrics"] = dm

    stages = raw.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("stages: non-empty list required")

    for i, st in enumerate(stages):
        if not isinstance(st, dict):
            raise TypeError(f"stages[{i}] must be a mapping")
        sp = st.get("sample")
        if not isinstance(sp, dict) or "n" not in sp or "seed" not in sp:
            raise KeyError(f"stages[{i}].sample needs n, seed")
        sp["n"] = int(sp["n"])
        sp["seed"] = int(sp["seed"])
        tr = st.get("train")
        if not isinstance(tr, dict) or "epochs" not in tr or "learning_rate" not in tr:
            raise KeyError(f"stages[{i}].train needs epochs, learning_rate")
        tr["epochs"] = int(tr["epochs"])
        tr["learning_rate"] = float(tr["learning_rate"])
        tr["weight_decay"] = float(tr.get("weight_decay", merged_defaults["weight_decay"]))
        if "batch_size" not in tr:
            tr["batch_size"] = int(merged_defaults["batch_size"])
        else:
            tr["batch_size"] = int(tr["batch_size"])

    return raw


def resolve_training_config_for_stage(spec: dict[str, Any], stage_index: int) -> dict[str, Any]:
    """
    Merge ``spec["defaults"]`` with ``spec["stages"][stage_index]`` top-level keys
    (excluding sample/train), then attach merged ``train`` from the stage.
    ``stage_index`` is 0-based (``stages[0]`` is training stage 1).
    """
    if stage_index < 0 or stage_index >= len(spec["stages"]):
        raise IndexError(f"stage_index {stage_index} out of range for stages")
    base = copy.deepcopy(spec["defaults"])
    st = spec["stages"][stage_index]
    override = {k: v for k, v in st.items() if k not in ("sample", "train")}
    merged = _deep_merge(base, override)
    tr = st["train"]
    merged["train"] = {
        "epochs": int(tr["epochs"]),
        "learning_rate": float(tr["learning_rate"]),
        "weight_decay": float(tr["weight_decay"]),
        "batch_size": int(tr["batch_size"]),
    }
    return merged


def spec_path_for_model(model_name: str) -> Path:
    for ext in (".yaml", ".yml"):
        p = MODEL_CONFIGS_DIR / f"{model_name}{ext}"
        if p.is_file():
            return p
    raise FileNotFoundError(f"No jepa2 spec at {MODEL_CONFIGS_DIR / (model_name + '.yaml')}")
