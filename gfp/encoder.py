"""Load a frozen BoardEncoderV3 from a jepa3 training checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from jepa3.architectures import resolve_config_for_id
from jepa3.architectures.chess_jepa_v3 import BoardEncoderV3
from jepa3.load import load_checkpoint_mapping


def _encoder_kwargs_from_architecture(
    architecture_id: str,
    architecture_config: dict[str, Any] | None,
) -> dict[str, Any]:
    cfg = resolve_config_for_id(architecture_id, architecture_config)
    return dict(
        d_model=int(cfg["d_model"]),
        n_layers=int(cfg["encoder_layers"]),
        nhead=int(cfg["nhead"]),
        dim_feedforward=int(cfg["dim_feedforward"]),
        dropout=float(cfg["dropout"]),
    )


def _strip_encoder_online_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "encoder_online."
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
    if not out:
        raise KeyError(
            "no keys with prefix 'encoder_online.' in checkpoint; "
            "expected a jepa3 ChessJEPAV3/V4 state dict"
        )
    return out


def load_jepa3_encoder_from_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device | None = None,
    strict: bool = True,
) -> BoardEncoderV3:
    """
    Load ``encoder_online`` weights into a standalone ``BoardEncoderV3`` and freeze it.

    Checkpoint must contain ``architecture_id``, ``architecture_config`` (optional),
    and ``model_state_dict`` as written by jepa3 training.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = load_checkpoint_mapping(path, map_location=dev)
    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint missing model_state_dict")
    arch_id = str(ckpt.get("architecture_id", "chess_jepa_v3"))
    arch_cfg = ckpt.get("architecture_config") or {}
    if not isinstance(arch_cfg, dict):
        raise TypeError("architecture_config must be a dict when present")

    enc_kw = _encoder_kwargs_from_architecture(arch_id, arch_cfg)
    encoder = BoardEncoderV3(**enc_kw)
    enc_sd = _strip_encoder_online_state(ckpt["model_state_dict"])
    encoder.load_state_dict(enc_sd, strict=strict)
    encoder.to(dev)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder
