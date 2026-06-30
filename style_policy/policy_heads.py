"""Pointer policy heads over the 64 square tokens. Heads return RAW logits; caller masks legality.

from-head: per-square score from its token (+ optional elo conditioning, + optional CLS token).
to-head:   per-target score conditioned on the chosen from-square token (+ optional elo, + optional CLS).

use_cls flag (default False): when True the d_model CLS vector is broadcast-concatenated to each of
the 64 square tokens before the score MLP (input dim grows by d_model). When False, cls= is ignored.

NOTE for eval scripts / onnx_export: those scripts run use_cls=False models and are unaffected because
cls defaults None and is ignored when use_cls=False. For a use_cls=True model, cls must be passed
from the encoder output — plumb it the same way as BasePolicy.forward_from/forward_to do.
"""
from __future__ import annotations
import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


class FromHead(nn.Module):
    def __init__(self, *, d_model: int, hidden: int, elo_dim: int = 0, n_elo_buckets: int = 0,
                 use_cls: bool = False):
        super().__init__()
        self.d_model = int(d_model)
        self.elo_dim = int(elo_dim)
        self.null_elo = int(n_elo_buckets)  # extra index for "unknown elo"
        self.use_cls = bool(use_cls)
        if elo_dim > 0:
            self.elo_emb = nn.Embedding(n_elo_buckets + 1, elo_dim)
        in_dim = d_model + self.elo_dim + (d_model if use_cls else 0)
        self.score = _mlp(in_dim, hidden, 1)

    def _elo_feat(self, b, device, elo_idx):
        if self.elo_dim == 0:
            return None
        if elo_idx is None:
            elo_idx = torch.full((b,), self.null_elo, device=device, dtype=torch.long)
        return self.elo_emb(elo_idx)  # (B, elo_dim)

    def forward(self, squares: torch.Tensor, *, cls: torch.Tensor | None = None,
                elo_idx: torch.Tensor | None = None) -> torch.Tensor:
        b = squares.shape[0]
        parts = [squares]
        feat = self._elo_feat(b, squares.device, elo_idx)
        if feat is not None:
            parts.append(feat.unsqueeze(1).expand(b, 64, self.elo_dim))
        if self.use_cls:
            if cls is None:
                raise ValueError("FromHead.forward: cls must be provided when use_cls=True")
            parts.append(cls.unsqueeze(1).expand(b, 64, self.d_model))
        return self.score(torch.cat(parts, dim=-1)).squeeze(-1)  # (B,64)


class ToHead(nn.Module):
    def __init__(self, *, d_model: int, hidden: int, elo_dim: int = 0, n_elo_buckets: int = 0,
                 use_cls: bool = False):
        super().__init__()
        self.d_model = int(d_model)
        self.elo_dim = int(elo_dim)
        self.null_elo = int(n_elo_buckets)
        self.use_cls = bool(use_cls)
        if elo_dim > 0:
            self.elo_emb = nn.Embedding(n_elo_buckets + 1, elo_dim)
        # target token concatenated with the chosen origin token + optional elo + optional cls
        in_dim = 2 * d_model + self.elo_dim + (d_model if use_cls else 0)
        self.score = _mlp(in_dim, hidden, 1)

    def forward(self, squares: torch.Tensor, from_sq: torch.Tensor, *,
                cls: torch.Tensor | None = None,
                elo_idx: torch.Tensor | None = None) -> torch.Tensor:
        b, _, d = squares.shape
        origin = squares[torch.arange(b, device=squares.device), from_sq.long()]  # (B,d)
        origin = origin.unsqueeze(1).expand(b, 64, d)
        parts = [squares, origin]
        if self.elo_dim > 0:
            elo_idx = elo_idx if elo_idx is not None else torch.full((b,), self.null_elo, device=squares.device, dtype=torch.long)
            parts.append(self.elo_emb(elo_idx).unsqueeze(1).expand(b, 64, self.elo_dim))
        if self.use_cls:
            if cls is None:
                raise ValueError("ToHead.forward: cls must be provided when use_cls=True")
            parts.append(cls.unsqueeze(1).expand(b, 64, self.d_model))
        return self.score(torch.cat(parts, dim=-1)).squeeze(-1)  # (B,64)
