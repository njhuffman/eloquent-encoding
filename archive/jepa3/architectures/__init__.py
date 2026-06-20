from __future__ import annotations

from typing import Any

import torch.nn as nn

from jepa3.architectures.chess_jepa_v3 import (
    ARCHITECTURE_ID as V3_ID,
    ChessJEPAV3,
    ChessJEPAV3Builder,
    DEFAULT_ARCHITECTURE_CONFIG as V3_DEFAULT,
    resolve_architecture_config as resolve_v3_architecture_config,
)
from jepa3.architectures.chess_jepa_v4 import (
    ARCHITECTURE_ID as V4_ID,
    ChessJEPAV4,
    ChessJEPAV4Builder,
    DEFAULT_ARCHITECTURE_CONFIG as V4_DEFAULT,
    resolve_architecture_config as resolve_v4_architecture_config,
)

_REGISTRY: dict[str, Any] = {
    "chess_jepa_v3": ChessJEPAV3Builder,
    "chess_jepa_v4": ChessJEPAV4Builder,
}


def build_model(architecture_id: str, architecture_config: dict[str, Any] | None) -> nn.Module:
    b = _REGISTRY.get(architecture_id)
    if b is None:
        raise KeyError(f"Unknown jepa3 architecture_id: {architecture_id!r}")
    return b.build(architecture_config)


def resolve_config_for_id(architecture_id: str, architecture_config: dict[str, Any] | None) -> dict[str, Any]:
    if architecture_id == "chess_jepa_v3":
        return resolve_v3_architecture_config(architecture_config)
    if architecture_id == "chess_jepa_v4":
        return resolve_v4_architecture_config(architecture_config)
    raise KeyError(architecture_id)


__all__ = [
    "V3_ID",
    "ChessJEPAV3",
    "ChessJEPAV3Builder",
    "V3_DEFAULT",
    "V4_ID",
    "ChessJEPAV4",
    "ChessJEPAV4Builder",
    "V4_DEFAULT",
    "build_model",
    "resolve_config_for_id",
    "resolve_v3_architecture_config",
    "resolve_v4_architecture_config",
]
