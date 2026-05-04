"""YAML model spec for world_model: defaults + per-stage deep merge.

Loss coefficients (``jepa_patch_weight``, CE weights, label smoothing, vicreg) and target
``ema_momentum`` must be set on each ``stages[i]``; they are not read from ``defaults:``
and have no built-in defaults.

Patch JEPA uses a Transformer predictor (four action tokens + 64 patches); tunables live under
``architecture.config`` (``predictor_encoder_layers``, ``predictor_nhead``, ``predictor_dim_feedforward``, ``predictor_dropout``).

Optional reconstruction weights ``recon_piece_ce_weight``, ``recon_turn_ce_weight``, and
``recon_can_move_ce_weight`` (defaults ``0.0``): 18-way patch category CE, CLS turn CE, and
per-square legal-origin CE vs packed ``from_mask``.

Optional aux weights ``aux_board_recon_weight`` and ``aux_meta_weight`` may be set in
``defaults`` and/or per stage (default ``0.0``; legacy v4 hooks removed from the training loop).

Optional ``early_stop_from_sq_top1`` in ``(0, 1]``: stop training when
``train_from_sq_top1/100`` exceeds that value (epoch mean over training batches).

Legacy ``early_stop_joint_top1`` is rejected (to-square CE was removed).
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIGS_DIR = _REPO_ROOT / "world_model" / "model_configs"

# Loss scaling / VICReg coefficients must appear on each stages[i], not under defaults.
_STAGE_REQUIRED_LOSS_SCALARS = (
    "jepa_patch_weight",
    "from_sq_ce_weight",
    "sq_ce_label_smoothing",
)
_VICREG_REQUIRED_KEYS = ("inv_coef", "var_coef", "cov_coef", "std_target")
_DEFAULTS_FORBIDDEN_STAGE_KEYS = (*_STAGE_REQUIRED_LOSS_SCALARS, "vicreg", "ema_momentum")


def _forbid_stage_scoped_keys_in_yaml_defaults(defaults: Any) -> None:
    if not isinstance(defaults, dict):
        return
    bad = [k for k in _DEFAULTS_FORBIDDEN_STAGE_KEYS if k in defaults]
    if bad:
        raise ValueError(
            "defaults must not set stage-scoped training keys "
            f"({', '.join(bad)}). "
            "Set jepa_patch_weight, from_sq_ce_weight, sq_ce_label_smoothing, "
            "vicreg (inv_coef, var_coef, cov_coef, std_target), and ema_momentum on each stages[i]."
        )


def _require_stage_loss_and_ema(stage: dict[str, Any], index: int) -> None:
    if "ema_momentum" not in stage:
        raise KeyError(
            f"stages[{index}] must set 'ema_momentum' (target EMA; required per stage, not in defaults)"
        )
    for k in _STAGE_REQUIRED_LOSS_SCALARS:
        if k not in stage:
            raise KeyError(
                f"stages[{index}] must set {k!r} (loss coefficients are required per stage, not in defaults)"
            )
    vr = stage.get("vicreg")
    if not isinstance(vr, dict):
        raise KeyError(
            f"stages[{index}].vicreg must be a mapping with "
            f"{', '.join(_VICREG_REQUIRED_KEYS)} (required per stage)"
        )
    for vk in _VICREG_REQUIRED_KEYS:
        if vk not in vr:
            raise KeyError(
                f"stages[{index}].vicreg must set {vk!r} (required per stage, no defaults)"
            )


def _reject_legacy_mse_played_weight(raw: dict[str, Any]) -> None:
    locs: list[str] = []

    def walk(obj: Any, prefix: str) -> None:
        if isinstance(obj, dict):
            if "mse_played_weight" in obj:
                locs.append(f"{prefix}.mse_played_weight" if prefix else "mse_played_weight")
            for k, v in obj.items():
                walk(v, f"{prefix}.{k}" if prefix else str(k))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{prefix}[{i}]")

    walk(raw, "")
    if locs:
        raise ValueError(
            "world_model model spec must not contain 'mse_played_weight'. "
            "Use vicreg.inv_coef instead. "
            f"Found at: {', '.join(sorted(set(locs)))}"
        )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


DEFAULTS: dict[str, Any] = {
    "from_sq_unknown_probability": 0.0,
    "batch_size": 256,
    "gradient_accumulation_steps": 1,
    "weight_decay": 0.05,
    "dataloader_num_workers": 0,
    "log_interval": 100,
    "train_log_mode": "compact",
    "max_gradient_norm": 0.0,
    "log_gradient_norms": True,
    "early_stop_from_sq_top1": None,
    "gsnr_probe_k": 8,
    "gsnr_probe_every_opt_steps": 0,
    "sam_rho": 0.0,
    "use_amp": True,
    "val_legal_seed": 42,
    "train_shuffle_seed": 0,
    "aux_board_recon_weight": 0.0,
    "aux_meta_weight": 0.0,
    "recon_piece_ce_weight": 0.0,
    "recon_turn_ce_weight": 0.0,
    "recon_can_move_ce_weight": 0.0,
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

    _reject_legacy_mse_played_weight(raw)

    ckpt = raw.get("checkpoint_dir")
    if not ckpt:
        raw["checkpoint_dir"] = str((_REPO_ROOT / "world_model_checkpoints" / name).resolve())
    else:
        p = Path(ckpt)
        raw["checkpoint_dir"] = str(p.expanduser().resolve() if p.is_absolute() else (_REPO_ROOT / p).resolve())

    _forbid_stage_scoped_keys_in_yaml_defaults(raw.get("defaults"))
    merged_defaults = _deep_merge(copy.deepcopy(DEFAULTS), raw.get("defaults") or {})
    raw["defaults"] = merged_defaults

    dm = _deep_merge(
        {
            "move_benchmark_sample_n": 10000,
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
        gas = int(tr.get("gradient_accumulation_steps", merged_defaults.get("gradient_accumulation_steps", 1)))
        if gas < 1:
            raise ValueError(f"stages[{i}].train.gradient_accumulation_steps must be >= 1 (got {gas})")
        tr["gradient_accumulation_steps"] = gas

        _require_stage_loss_and_ema(st, i)

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
    _gas = int(merged["train"]["gradient_accumulation_steps"])
    if _gas < 1:
        raise ValueError(f"train.gradient_accumulation_steps must be >= 1 (got {_gas})")

    for wkey in ("jepa_patch_weight", "from_sq_ce_weight"):
        wv = float(merged[wkey])
        if not math.isfinite(wv) or wv < 0.0:
            raise ValueError(f"{wkey} must be finite and >= 0 (got {merged[wkey]!r})")

    sm = float(merged["sq_ce_label_smoothing"])
    if not math.isfinite(sm) or sm < 0.0 or sm > 1.0:
        raise ValueError(f"sq_ce_label_smoothing must be in [0, 1] (got {merged['sq_ce_label_smoothing']!r})")

    p_unk = float(merged.get("from_sq_unknown_probability", 0.0))
    if not math.isfinite(p_unk) or p_unk < 0.0 or p_unk > 1.0:
        raise ValueError(
            "from_sq_unknown_probability must be finite and in [0, 1] "
            f"(got {merged.get('from_sq_unknown_probability')!r})"
        )

    vr = merged["vicreg"]
    if not isinstance(vr, dict):
        raise TypeError("vicreg must be a mapping")
    for key in ("inv_coef", "var_coef", "cov_coef"):
        v = float(vr[key])
        if not math.isfinite(v) or v < 0.0:
            raise ValueError(f"vicreg.{key} must be finite and >= 0 (got {vr[key]!r})")
    stt = float(vr["std_target"])
    if not math.isfinite(stt) or stt <= 0.0:
        raise ValueError(f"vicreg.std_target must be finite and > 0 (got {vr['std_target']!r})")

    ema_m = float(merged["ema_momentum"])
    if not math.isfinite(ema_m) or ema_m < 0.0 or ema_m > 1.0:
        raise ValueError(f"ema_momentum must be finite and in [0, 1] (got {merged['ema_momentum']!r})")

    gk = int(merged.get("gsnr_probe_k", 8))
    ge = int(merged.get("gsnr_probe_every_opt_steps", 0))
    if ge < 0:
        raise ValueError(f"gsnr_probe_every_opt_steps must be >= 0 (got {ge})")
    if ge > 0:
        if gk < 2:
            raise ValueError(f"gsnr_probe_k must be >= 2 when GSNR is enabled (got {gk})")
        gas = int(merged["train"]["gradient_accumulation_steps"])
        if gk > ge * max(gas, 1):
            raise ValueError(
                "gsnr_probe_k should not exceed gsnr_probe_every_opt_steps * "
                f"gradient_accumulation_steps; got k={gk}, every={ge}, accum={gas}"
            )
    sam_rho = float(merged.get("sam_rho", 0.0))
    if not math.isfinite(sam_rho) or sam_rho < 0.0:
        raise ValueError(f"sam_rho must be finite and >= 0 (got {merged.get('sam_rho')!r})")
    tlm = merged.get("train_log_mode", "compact")
    if tlm not in ("compact", "full"):
        raise ValueError(f'train_log_mode must be "compact" or "full" (got {tlm!r})')
    merged["train_log_mode"] = str(tlm)

    raw_mgn = merged.get("max_gradient_norm", 0.0)
    if raw_mgn is None:
        mgn = 0.0
    else:
        mgn = float(raw_mgn)
    if not math.isfinite(mgn) or mgn < 0.0:
        raise ValueError(f"max_gradient_norm must be finite and >= 0 (got {raw_mgn!r})")
    merged["max_gradient_norm"] = mgn

    lgn = merged.get("log_gradient_norms", True)
    if not isinstance(lgn, bool):
        raise ValueError(f"log_gradient_norms must be a bool (got {lgn!r})")
    merged["log_gradient_norms"] = bool(lgn)

    for pw in (
        "aux_board_recon_weight",
        "aux_meta_weight",
        "recon_piece_ce_weight",
        "recon_turn_ce_weight",
        "recon_can_move_ce_weight",
    ):
        wv = float(merged.get(pw, 0.0))
        if not math.isfinite(wv) or wv < 0.0:
            raise ValueError(f"{pw} must be finite and >= 0 (got {merged.get(pw)!r})")
        merged[pw] = wv

    if merged.get("early_stop_joint_top1") is not None:
        raise ValueError(
            "world_model no longer supports early_stop_joint_top1 (to-square CE removed). "
            "Use early_stop_from_sq_top1 in (0, 1] to stop on train_from_sq_top1 fraction."
        )

    raw_es = merged.get("early_stop_from_sq_top1")
    if raw_es is None:
        merged["early_stop_from_sq_top1"] = None
    else:
        x = float(raw_es)
        if not math.isfinite(x) or x <= 0.0 or x > 1.0:
            raise ValueError(
                "early_stop_from_sq_top1 must be in (0, 1] when set "
                f"(train_from_sq_top1/100 must exceed this to stop; got {raw_es!r})"
            )
        merged["early_stop_from_sq_top1"] = x

    return merged


def spec_path_for_model(model_name: str) -> Path:
    for ext in (".yaml", ".yml"):
        p = MODEL_CONFIGS_DIR / f"{model_name}{ext}"
        if p.is_file():
            return p
    raise FileNotFoundError(f"No world_model spec at {MODEL_CONFIGS_DIR / (model_name + '.yaml')}")
