"""
Registered Chess-JEPA architecture builders.
"""

from __future__ import annotations

from .chess_jepa_v1 import (
    ARCHITECTURE_ID as CHESS_JEPA_V1,
    ChessJEPA,
    ChessJEPABuilder,
    DEFAULT_ARCHITECTURE_CONFIG,
    jepa_triplet_vicreg_loss,
    resolve_architecture_config as resolve_chess_jepa_config,
)

DEFAULT_ARCHITECTURE_ID = CHESS_JEPA_V1

ARCHITECTURE_BUILDERS: dict[str, type] = {
    CHESS_JEPA_V1: ChessJEPABuilder,
}


def build_model(architecture_id: str, architecture_config: dict | None = None) -> ChessJEPA:
    if architecture_id not in ARCHITECTURE_BUILDERS:
        known = ", ".join(sorted(ARCHITECTURE_BUILDERS))
        raise ValueError(f"Unknown architecture_id={architecture_id!r}. Known: {known}")
    builder = ARCHITECTURE_BUILDERS[architecture_id]
    return builder.build(architecture_config)


def resolve_config_for_id(architecture_id: str, architecture_config: dict | None) -> dict:
    if architecture_id not in ARCHITECTURE_BUILDERS:
        known = ", ".join(sorted(ARCHITECTURE_BUILDERS))
        raise ValueError(f"Unknown architecture_id={architecture_id!r}. Known: {known}")
    if architecture_id == CHESS_JEPA_V1:
        return dict(resolve_chess_jepa_config(architecture_config))
    return dict(architecture_config or {})


__all__ = [
    "ARCHITECTURE_BUILDERS",
    "CHESS_JEPA_V1",
    "ChessJEPA",
    "ChessJEPABuilder",
    "DEFAULT_ARCHITECTURE_CONFIG",
    "DEFAULT_ARCHITECTURE_ID",
    "build_model",
    "jepa_triplet_vicreg_loss",
    "resolve_config_for_id",
]
