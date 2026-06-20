"""YAML model spec for gfp: defaults + per-stage deep merge (jepa3-style)."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIGS_DIR = _REPO_ROOT / "gfp" / "model_configs"

GFP_ARCHITECTURE_ID = "gfp_from_mlp"

# Per-stage only (not in defaults), like jepa3 CE smoothing on each stage.
_STAGE_FORBIDDEN_IN_DEFAULTS = ("sq_ce_label_smoothing",)


def _forbid_stage_keys_in_defaults(defaults: Any) -> None:
    if not isinstance(defaults, dict):
        return
    bad = [k for k in _STAGE_FORBIDDEN_IN_DEFAULTS if k in defaults]
    if bad:
        raise ValueError(
            "defaults must not set stage-scoped keys "
            f"({', '.join(bad)}). Set sq_ce_label_smoothing on each stages[i]."
        )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_repo_path(raw: str | Path) -> Path:
    p = Path(raw)
    if not p.parts:
        raise ValueError("path is empty")
    return p.expanduser().resolve() if p.is_absolute() else (_REPO_ROOT / p).resolve()


DEFAULTS: dict[str, Any] = {
    "batch_size": 256,
    "dataloader_num_workers": 4,
    "train_shuffle_seed": 0,
    "use_amp": True,
    "weight_decay": 0.01,
    "max_gradient_norm": 0.0,
    "log_interval": 100,
    "encoder_strict": True,
    "seed": 0,
    "early_stop_train_top1": None,
    "gradient_accumulation_steps": 1,
}


def load_model_spec(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("model spec root must be a mapping")
    name = raw.get("name")
    if not name or not isinstance(name, str):
        raise ValueError('spec must have string "name" (same as --model)')
    raw["name"] = name.strip()

    enc = raw.get("encoder_checkpoint")
    if not enc or not isinstance(enc, str):
        raise ValueError("encoder_checkpoint required (path to jepa3 .pt with encoder_online weights)")
    raw["encoder_checkpoint"] = str(_resolve_repo_path(enc))

    for key in ("train_dataset_h5", "val_dataset_h5"):
        p = raw.get(key)
        if not p or not isinstance(p, str):
            raise ValueError(f"{key} required (path to gfp HDF5)")
        raw[key] = str(_resolve_repo_path(p))

    vs = raw.get("val_sample")
    if not isinstance(vs, dict) or "n" not in vs or "seed" not in vs:
        raise ValueError("val_sample: {n, seed} required")
    vs["n"] = int(vs["n"])
    vs["seed"] = int(vs["seed"])
    raw["val_sample"] = vs

    ckpt = raw.get("checkpoint_dir")
    if not ckpt:
        raw["checkpoint_dir"] = str((_REPO_ROOT / "gfp_checkpoints" / raw["name"]).resolve())
    else:
        raw["checkpoint_dir"] = str(_resolve_repo_path(ckpt))

    arch = raw.get("architecture")
    if not isinstance(arch, dict) or not arch.get("id"):
        raise ValueError('spec must have "architecture" with id')
    if str(arch["id"]) != GFP_ARCHITECTURE_ID:
        raise ValueError(f'architecture.id must be {GFP_ARCHITECTURE_ID!r} (got {arch.get("id")!r})')
    cfg = arch.get("config")
    if not isinstance(cfg, dict):
        raise ValueError("architecture.config must be a mapping")
    for k in ("head_hidden", "head_depth"):
        if k not in cfg:
            raise KeyError(f"architecture.config must set {k!r}")
    cfg["head_hidden"] = int(cfg["head_hidden"])
    cfg["head_depth"] = int(cfg["head_depth"])
    if cfg["head_hidden"] < 1 or cfg["head_depth"] < 1:
        raise ValueError("head_hidden and head_depth must be >= 1")

    _forbid_stage_keys_in_defaults(raw.get("defaults"))
    merged_defaults = _deep_merge(copy.deepcopy(DEFAULTS), raw.get("defaults") or {})
    raw["defaults"] = merged_defaults

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
        gas = int(tr.get("gradient_accumulation_steps", merged_defaults.get("gradient_accumulation_steps", 1)))
        if gas < 1:
            raise ValueError(f"stages[{i}].train.gradient_accumulation_steps must be >= 1 (got {gas})")
        tr["gradient_accumulation_steps"] = gas

        if "sq_ce_label_smoothing" not in st:
            raise KeyError(f"stages[{i}] must set sq_ce_label_smoothing (per stage, not in defaults)")
        sm = float(st["sq_ce_label_smoothing"])
        if not math.isfinite(sm) or sm < 0.0 or sm > 1.0:
            raise ValueError(f"stages[{i}].sq_ce_label_smoothing must be in [0, 1] (got {st['sq_ce_label_smoothing']!r})")

    return raw


def resolve_training_config_for_stage(spec: dict[str, Any], stage_index: int) -> dict[str, Any]:
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
        "gradient_accumulation_steps": int(
            tr.get("gradient_accumulation_steps", merged.get("gradient_accumulation_steps", 1))
        ),
    }
    gas = int(merged["train"]["gradient_accumulation_steps"])
    if gas < 1:
        raise ValueError(f"train.gradient_accumulation_steps must be >= 1 (got {gas})")

    sm = float(merged["sq_ce_label_smoothing"])
    if not math.isfinite(sm) or sm < 0.0 or sm > 1.0:
        raise ValueError(f"sq_ce_label_smoothing must be in [0, 1] (got {merged['sq_ce_label_smoothing']!r})")

    raw_mgn = merged.get("max_gradient_norm", 0.0)
    mgn = 0.0 if raw_mgn is None else float(raw_mgn)
    if not math.isfinite(mgn) or mgn < 0.0:
        raise ValueError(f"max_gradient_norm must be finite and >= 0 (got {raw_mgn!r})")
    merged["max_gradient_norm"] = mgn

    raw_es = merged.get("early_stop_train_top1")
    if raw_es is None:
        merged["early_stop_train_top1"] = None
    else:
        x = float(raw_es)
        if not math.isfinite(x) or x <= 0.0 or x > 1.0:
            raise ValueError(
                "early_stop_train_top1 must be in (0, 1] when set "
                f"(stop when train from-sq top1% / 100 exceeds this; got {raw_es!r})"
            )
        merged["early_stop_train_top1"] = x

    return merged


def spec_path_for_model(model_name: str) -> Path:
    for ext in (".yaml", ".yml"):
        p = MODEL_CONFIGS_DIR / f"{model_name}{ext}"
        if p.is_file():
            return p
    raise FileNotFoundError(f"No gfp spec at {MODEL_CONFIGS_DIR / (model_name + '.yaml')}")
