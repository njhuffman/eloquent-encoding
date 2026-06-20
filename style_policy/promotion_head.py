"""Promotion head: 4-way (knight,bishop,rook,queen). Fired only on pawn-to-back-rank moves."""
from __future__ import annotations
import torch
import torch.nn as nn


class PromotionHead(nn.Module):
    def __init__(self, *, d_model: int, hidden: int = 64):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(2 * d_model, hidden), nn.GELU(), nn.Linear(hidden, 4))

    def forward(self, squares: torch.Tensor, from_sq: torch.Tensor, to_sq: torch.Tensor) -> torch.Tensor:
        b = squares.shape[0]
        idx = torch.arange(b, device=squares.device)
        feat = torch.cat([squares[idx, from_sq.long()], squares[idx, to_sq.long()]], dim=-1)
        return self.score(feat)  # (B,4)
