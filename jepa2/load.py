"""Load jepa2 models from checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from jepa2.architectures import build_model


def load_checkpoint_mapping(
    path: str | Path,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    if map_location is None:
        map_location = "cpu"
    return torch.load(path, map_location=map_location, weights_only=False)


def model_from_checkpoint_state(
    state: dict[str, Any],
    *,
    map_location: str | torch.device | None = None,
    strict: bool = True,
) -> nn.Module:
    if "model_state_dict" not in state:
        raise KeyError("Checkpoint missing model_state_dict")
    arch_id = state.get("architecture_id", "chess_jepa_v2")
    arch_cfg = state.get("architecture_config") or {}
    model = build_model(arch_id, arch_cfg)
    model.load_state_dict(state["model_state_dict"], strict=strict)
    return model


def load_jepa2_from_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device | None = None,
    strict: bool = True,
) -> nn.Module:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = load_checkpoint_mapping(path, map_location=dev)
    model = model_from_checkpoint_state(state, map_location=dev, strict=strict)
    return model.to(dev)
