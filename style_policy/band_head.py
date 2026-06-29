"""Per-band specialized heads on a frozen encoder (elo-agnostic conditioning by hard band split)."""
from __future__ import annotations
import torch
import torch.nn as nn
from style_policy.policy_heads import FromHead, ToHead

class BandHead(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.from_head = FromHead(d_model=d_model, hidden=hidden, elo_dim=0)
        self.to_head = ToHead(d_model=d_model, hidden=hidden, elo_dim=0)

    def from_logits(self, squares: torch.Tensor) -> torch.Tensor:
        return self.from_head(squares)

    def to_logits(self, squares: torch.Tensor, from_sq: torch.Tensor) -> torch.Tensor:
        return self.to_head(squares, from_sq)
