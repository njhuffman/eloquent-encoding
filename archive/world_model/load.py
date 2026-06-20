"""Load world_model models from checkpoints.

Checkpoints are tied to the architecture graph in the saving code; loading with ``strict=True``
may fail after structural changes (e.g. replacing move heads). Re-run stage 0 init when needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from world_model.architectures import build_model


def load_checkpoint_mapping(
    path: str | Path,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    if map_location is None:
        map_location = "cpu"
    return torch.load(path, map_location=map_location, weights_only=False)


def world_model_architecture_fields_from_checkpoint(state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Require explicit ``architecture_id`` and ``architecture_config`` (no legacy defaults)."""
    aid = state.get("architecture_id")
    if not isinstance(aid, str) or not aid.strip():
        raise KeyError("Checkpoint missing non-empty string architecture_id")
    if "architecture_config" not in state:
        raise KeyError("Checkpoint missing architecture_config")
    cfg = state["architecture_config"]
    if not isinstance(cfg, dict):
        raise TypeError("architecture_config must be a dict")
    return aid.strip(), cfg


def model_from_checkpoint_state(state: dict[str, Any]) -> nn.Module:
    if "model_state_dict" not in state:
        raise KeyError("Checkpoint missing model_state_dict")
    arch_id, arch_cfg = world_model_architecture_fields_from_checkpoint(state)
    model = build_model(arch_id, arch_cfg)
    model.load_state_dict(state["model_state_dict"], strict=True)
    return model


def load_world_model_from_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device | None = None,
) -> nn.Module:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = load_checkpoint_mapping(path, map_location=dev)
    model = model_from_checkpoint_state(state)
    return model.to(dev)
