"""Pointer policy heads over the 64 square tokens. Heads return RAW logits; caller masks legality.

from-head: per-square score from its token (+ optional elo conditioning).
to-head:   per-target score conditioned on the chosen from-square token (+ optional elo).
"""
from __future__ import annotations
import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


class FromHead(nn.Module):
    def __init__(self, *, d_model: int, hidden: int, elo_dim: int = 0, n_elo_buckets: int = 0):
        super().__init__()
        self.elo_dim = int(elo_dim)
        self.null_elo = int(n_elo_buckets)  # extra index for "unknown elo"
        if elo_dim > 0:
            self.elo_emb = nn.Embedding(n_elo_buckets + 1, elo_dim)
        self.score = _mlp(d_model + self.elo_dim, hidden, 1)

    def _elo_feat(self, b, device, elo_idx):
        if self.elo_dim == 0:
            return None
        if elo_idx is None:
            elo_idx = torch.full((b,), self.null_elo, device=device, dtype=torch.long)
        return self.elo_emb(elo_idx)  # (B, elo_dim)

    def forward(self, squares: torch.Tensor, *, elo_idx: torch.Tensor | None = None) -> torch.Tensor:
        b = squares.shape[0]
        feat = self._elo_feat(b, squares.device, elo_idx)
        if feat is not None:
            squares = torch.cat([squares, feat.unsqueeze(1).expand(b, 64, self.elo_dim)], dim=-1)
        return self.score(squares).squeeze(-1)  # (B,64)


class ToHead(nn.Module):
    def __init__(self, *, d_model: int, hidden: int, elo_dim: int = 0, n_elo_buckets: int = 0):
        super().__init__()
        self.elo_dim = int(elo_dim)
        self.null_elo = int(n_elo_buckets)
        if elo_dim > 0:
            self.elo_emb = nn.Embedding(n_elo_buckets + 1, elo_dim)
        # target token concatenated with the chosen origin token + optional elo
        self.score = _mlp(2 * d_model + self.elo_dim, hidden, 1)

    def forward(self, squares: torch.Tensor, from_sq: torch.Tensor, *, elo_idx: torch.Tensor | None = None) -> torch.Tensor:
        b, _, d = squares.shape
        origin = squares[torch.arange(b, device=squares.device), from_sq.long()]  # (B,d)
        origin = origin.unsqueeze(1).expand(b, 64, d)
        parts = [squares, origin]
        if self.elo_dim > 0:
            elo_idx = elo_idx if elo_idx is not None else torch.full((b,), self.null_elo, device=squares.device, dtype=torch.long)
            parts.append(self.elo_emb(elo_idx).unsqueeze(1).expand(b, 64, self.elo_dim))
        return self.score(torch.cat(parts, dim=-1)).squeeze(-1)  # (B,64)
