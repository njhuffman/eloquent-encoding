"""Global-from MLP on frozen encoder global embedding."""

from __future__ import annotations

import torch
import torch.nn as nn

from jepa3.architectures.chess_jepa_v3 import BoardEncoderV3


def _mlp_gelu(in_dim: int, hidden: int, depth: int, out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = int(in_dim)
    h = int(hidden)
    for _ in range(int(depth) - 1):
        layers.append(nn.Linear(d, h))
        layers.append(nn.GELU())
        d = h
    layers.append(nn.Linear(d, int(out_dim)))
    return nn.Sequential(*layers)


class FromSquareMlpHead(nn.Module):
    """64-way from-square logits from global embedding (same role as jepa3 FromSquareHead, gfp-owned)."""

    def __init__(self, d_model: int, hidden: int, depth: int) -> None:
        super().__init__()
        self.net = _mlp_gelu(d_model, hidden, depth, 64)

    def forward(self, z_global: torch.Tensor) -> torch.Tensor:
        return self.net(z_global)


class GlobalFromPredictor(nn.Module):
    """Frozen jepa3-style board encoder + trainable from-square head."""

    def __init__(
        self,
        encoder: BoardEncoderV3,
        *,
        head_hidden: int = 512,
        head_depth: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = FromSquareMlpHead(int(encoder.d_model), int(head_hidden), int(head_depth))

    def train(self, mode: bool = True) -> GlobalFromPredictor:
        super().train(mode)
        self.encoder.eval()
        return self

    def trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.head.parameters())

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.encoder(board)
        return self.head(z)
