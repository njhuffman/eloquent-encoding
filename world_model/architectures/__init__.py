from __future__ import annotations

from typing import Any

import torch.nn as nn

from world_model.architectures.chess_world_model_v1 import (
    ARCHITECTURE_ID as WM_V1_ID,
    ChessWorldModelV1,
    ChessWorldModelV1Builder,
    DEFAULT_ARCHITECTURE_CONFIG as WM_V1_DEFAULT,
    resolve_architecture_config as resolve_wm_v1_architecture_config,
)

_REGISTRY: dict[str, Any] = {
    "chess_world_model_v1": ChessWorldModelV1Builder,
}


def build_model(architecture_id: str, architecture_config: dict[str, Any] | None) -> nn.Module:
    b = _REGISTRY.get(architecture_id)
    if b is None:
        raise KeyError(f"Unknown world_model architecture_id: {architecture_id!r}")
    return b.build(architecture_config)


def resolve_config_for_id(architecture_id: str, architecture_config: dict[str, Any] | None) -> dict[str, Any]:
    if architecture_id == "chess_world_model_v1":
        return resolve_wm_v1_architecture_config(architecture_config)
    raise KeyError(architecture_id)


__all__ = [
    "WM_V1_ID",
    "ChessWorldModelV1",
    "ChessWorldModelV1Builder",
    "WM_V1_DEFAULT",
    "build_model",
    "resolve_config_for_id",
    "resolve_wm_v1_architecture_config",
]
