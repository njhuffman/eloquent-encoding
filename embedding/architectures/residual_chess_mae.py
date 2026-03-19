"""
Residual CNN MAE (current default). architecture_config is merged with DEFAULT_ARCHITECTURE_CONFIG.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from embedding.model import ChessMAE

ARCHITECTURE_ID = "residual_chess_mae_v1"

# JSON-serializable defaults (lists not tuples for merge output)
DEFAULT_ARCHITECTURE_CONFIG: dict[str, Any] = {
    "embedding_dim": 128,
    "stem_channels": 128,
    "num_res_blocks_low": 2,
    "mid_channels": 256,
    "num_res_blocks_high": 1,
    "mlp_hidden": 1024,
    "dropout": 0.2,
    "decoder_channels": [256, 128, 64],
}


def resolve_architecture_config(architecture_config: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_ARCHITECTURE_CONFIG)
    if architecture_config:
        merged.update(architecture_config)
    # Normalize decoder_channels to list of int
    dc = merged["decoder_channels"]
    merged["decoder_channels"] = [int(x) for x in dc]
    return merged


class ResidualChessMAEBuilder:
    """Builds `ChessMAE` from architecture_config for ARCHITECTURE_ID."""

    architecture_id = ARCHITECTURE_ID

    @staticmethod
    def resolve_config(architecture_config: dict[str, Any] | None) -> dict[str, Any]:
        return resolve_architecture_config(architecture_config)

    @staticmethod
    def build(architecture_config: dict[str, Any] | None = None) -> ChessMAE:
        c = resolve_architecture_config(architecture_config)
        dec = tuple(c["decoder_channels"])
        return ChessMAE(
            embedding_dim=int(c["embedding_dim"]),
            stem_channels=int(c["stem_channels"]),
            num_res_blocks_low=int(c["num_res_blocks_low"]),
            mid_channels=int(c["mid_channels"]),
            num_res_blocks_high=int(c["num_res_blocks_high"]),
            mlp_hidden=int(c["mlp_hidden"]),
            dropout=float(c["dropout"]),
            decoder_channels=dec,
        )

    @staticmethod
    def build_module(architecture_config: dict[str, Any] | None = None) -> nn.Module:
        return ResidualChessMAEBuilder.build(architecture_config)
