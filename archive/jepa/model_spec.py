"""YAML model spec: architecture, train/val move HDF5 paths, per-stage sample/train."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIGS_DIR = _REPO_ROOT / "jepa" / "model_configs"


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
    "triplet_margin_alpha": 0.2,
    "vicreg_var_coef": 0.1,
    "vicreg_std_target": 1.0,
    "batch_size": 256,
    "weight_decay": 0.05,
    "dataloader_num_workers": 4,
    "log_interval": 100,
    "use_amp": True,
}

DASHBOARD_METRICS_DEFAULTS: dict[str, Any] = {
    "move_benchmark_sample_n": 2048,
    "move_benchmark_seed": 42,
    "move_benchmark_train_seed": 1000045,
    "move_benchmark_succ_chunk": 256,
    "device": "auto",
}


def load_model_spec(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("model spec root must be a mapping")
    name = raw.get("name")
    if not name or not isinstance(name, str):
        raise ValueError('spec must have string "name" (same as --model)')
    name = name.strip()
    raw["name"] = name

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

    ckpt = raw.get("checkpoint_dir")
    if not ckpt:
        raw["checkpoint_dir"] = str((_REPO_ROOT / "jepa_checkpoints" / name).resolve())
    else:
        p = Path(ckpt)
        raw["checkpoint_dir"] = str(p.expanduser().resolve() if p.is_absolute() else (_REPO_ROOT / p).resolve())

    cd = raw.get("cache_dir")
    if cd is None:
        raw["cache_dir"] = str(Path(raw["checkpoint_dir"]) / "cache")
    else:
        p = Path(cd)
        raw["cache_dir"] = str(p.expanduser().resolve() if p.is_absolute() else (_REPO_ROOT / p).resolve())

    stages = raw.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("stages: non-empty list required for training stages")

    merged_defaults = _deep_merge(DEFAULTS, raw.get("defaults") or {})
    merged_defaults.pop("in_memory", None)
    raw["defaults"] = merged_defaults

    dm = _deep_merge(dict(DASHBOARD_METRICS_DEFAULTS), raw.get("dashboard_metrics") or {})
    dm["move_benchmark_sample_n"] = int(dm["move_benchmark_sample_n"])
    dm["move_benchmark_seed"] = int(dm["move_benchmark_seed"])
    dm["move_benchmark_train_seed"] = int(dm["move_benchmark_train_seed"])
    dm["move_benchmark_succ_chunk"] = int(dm["move_benchmark_succ_chunk"])
    if dm["device"] not in ("auto", "cuda", "cpu"):
        raise ValueError('dashboard_metrics.device must be "auto", "cuda", or "cpu"')
    raw["dashboard_metrics"] = dm

    for i, st in enumerate(stages):
        if not isinstance(st, dict):
            raise TypeError(f"stages[{i}] must be a mapping")
        sp = st.get("sample")
        if not isinstance(sp, dict) or "n" not in sp or "seed" not in sp:
            raise KeyError(f"stages[{i}].sample needs n, seed")
        sp["n"] = int(sp["n"])
        sp["seed"] = int(sp["seed"])
        hn = st.get("hard_negatives")
        if not isinstance(hn, dict) or "n_hard" not in hn or "m_random" not in hn:
            raise KeyError(f"stages[{i}].hard_negatives needs n_hard, m_random")
        hn["n_hard"] = int(hn["n_hard"])
        hn["m_random"] = int(hn["m_random"])
        if hn["n_hard"] < 0 or hn["m_random"] < 0:
            raise ValueError(f"stages[{i}].hard_negatives n_hard and m_random must be >= 0")
        if hn["n_hard"] + hn["m_random"] < 1:
            raise ValueError(
                f"stages[{i}].hard_negatives n_hard + m_random must be >= 1 (got "
                f"{hn['n_hard']}+{hn['m_random']})"
            )
        if "evaluate_legals_n" in hn:
            ev = int(hn["evaluate_legals_n"])
            if ev < 2:
                raise ValueError(f"stages[{i}].hard_negatives.evaluate_legals_n must be >= 2, got {ev}")
            hn["evaluate_legals_n"] = ev
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


def spec_path_for_model(model_name: str) -> Path:
    for ext in (".yaml", ".yml", ".json"):
        p = MODEL_CONFIGS_DIR / f"{model_name}{ext}"
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"No spec at {MODEL_CONFIGS_DIR / (model_name + '.yaml')}"
        f" (or .yml / .json)"
    )
