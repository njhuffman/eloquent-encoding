"""
Registered MAE architecture families. Each builder maps architecture_config -> nn.Module
with forward(encoder_input, mask) -> (embedding, piece_logits) and an .encoder attribute.
"""

from __future__ import annotations

from .residual_chess_mae import (
    ARCHITECTURE_ID as RESIDUAL_CHESS_MAE_V1,
    DEFAULT_ARCHITECTURE_CONFIG,
    ResidualChessMAEBuilder,
    resolve_architecture_config as resolve_residual_config,
)

DEFAULT_ARCHITECTURE_ID = RESIDUAL_CHESS_MAE_V1

ARCHITECTURE_BUILDERS: dict[str, type] = {
    RESIDUAL_CHESS_MAE_V1: ResidualChessMAEBuilder,
}


def build_model(architecture_id: str, architecture_config: dict | None = None) -> nn.Module:
    if architecture_id not in ARCHITECTURE_BUILDERS:
        known = ", ".join(sorted(ARCHITECTURE_BUILDERS))
        raise ValueError(f"Unknown architecture_id={architecture_id!r}. Known: {known}")
    builder = ARCHITECTURE_BUILDERS[architecture_id]
    return builder.build(architecture_config)


def resolve_config_for_id(architecture_id: str, architecture_config: dict | None) -> dict:
    if architecture_id not in ARCHITECTURE_BUILDERS:
        known = ", ".join(sorted(ARCHITECTURE_BUILDERS))
        raise ValueError(f"Unknown architecture_id={architecture_id!r}. Known: {known}")
    builder = ARCHITECTURE_BUILDERS[architecture_id]
    if hasattr(builder, "resolve_config"):
        return dict(builder.resolve_config(architecture_config))
    return dict(architecture_config or {})


__all__ = [
    "ARCHITECTURE_BUILDERS",
    "DEFAULT_ARCHITECTURE_CONFIG",
    "DEFAULT_ARCHITECTURE_ID",
    "RESIDUAL_CHESS_MAE_V1",
    "ResidualChessMAEBuilder",
    "build_model",
    "resolve_config_for_id",
    "resolve_residual_config",
]
