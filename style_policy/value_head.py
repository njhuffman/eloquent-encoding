"""WDL value head over the encoder CLS token (+ optional mover-elo conditioning).

Returns RAW 3-logit win/draw/loss scores in the order (loss=0, draw=1, win=2),
matching the `result` label encoding. Mirrors the elo-embedding pattern in
policy_heads.py (own embedding, extra index for unknown elo)."""
from __future__ import annotations
import torch
import torch.nn as nn


class WDLHead(nn.Module):
    def __init__(self, *, d_model: int, hidden: int, elo_dim: int = 0, n_elo_buckets: int = 0):
        super().__init__()
        self.elo_dim = int(elo_dim)
        self.null_elo = int(n_elo_buckets)
        if elo_dim > 0:
            self.elo_emb = nn.Embedding(n_elo_buckets + 1, elo_dim)
        self.score = nn.Sequential(
            nn.Linear(d_model + self.elo_dim, hidden), nn.GELU(), nn.Linear(hidden, 3))

    def forward(self, cls: torch.Tensor, *, elo_idx: torch.Tensor | None = None) -> torch.Tensor:
        b = cls.shape[0]
        if self.elo_dim > 0:
            if elo_idx is None:
                elo_idx = torch.full((b,), self.null_elo, device=cls.device, dtype=torch.long)
            cls = torch.cat([cls, self.elo_emb(elo_idx)], dim=-1)
        return self.score(cls)  # (B,3)
