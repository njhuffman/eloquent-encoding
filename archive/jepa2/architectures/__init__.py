from __future__ import annotations

from typing import Any

import torch.nn as nn

from jepa2.architectures.chess_jepa_v2 import (
    ARCHITECTURE_ID,
    ChessJEPA,
    ChessJEPABuilder,
    DEFAULT_ARCHITECTURE_CONFIG,
    resolve_architecture_config,
)
from jepa2.architectures.chess_jepa_v2_final_ln import ChessJEPAFinalLnBuilder
from jepa2.architectures.chess_jepa_v2_fromsq_ln import ChessJEPAFromSqLnBuilder

_REGISTRY: dict[str, Any] = {
    "chess_jepa_v2": ChessJEPABuilder,
    "chess_jepa_v2_final_ln": ChessJEPAFinalLnBuilder,
    "chess_jepa_v2_fromsq_ln": ChessJEPAFromSqLnBuilder,
}


def build_model(architecture_id: str, architecture_config: dict[str, Any] | None) -> nn.Module:
    b = _REGISTRY.get(architecture_id)
    if b is None:
        raise KeyError(f"Unknown jepa2 architecture_id: {architecture_id!r}")
    return b.build(architecture_config)


def resolve_config_for_id(architecture_id: str, architecture_config: dict[str, Any] | None) -> dict[str, Any]:
    if architecture_id in ("chess_jepa_v2", "chess_jepa_v2_final_ln", "chess_jepa_v2_fromsq_ln"):
        return resolve_architecture_config(architecture_config)
    raise KeyError(architecture_id)
