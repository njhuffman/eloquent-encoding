"""Query/key pointer policy head: joint P(from,to) via sum_h q_i^h . k_j^h over 64 square tokens.
Contrast with the factored FromHead/ToHead. Promotion piece is NOT modeled (probe = from/to only).
Optional CLS injected as a global conditioner added to every square token before the projections."""
from __future__ import annotations
import torch
import torch.nn as nn

class PointerHead(nn.Module):
    def __init__(self, d_model: int, d_head: int = 64, n_heads: int = 1, use_cls: bool = True):
        super().__init__()
        self.n_heads = int(n_heads); self.d_head = int(d_head); self.use_cls = bool(use_cls)
        self.q_proj = nn.Linear(d_model, n_heads * d_head)
        self.k_proj = nn.Linear(d_model, n_heads * d_head)
        if use_cls:
            self.cls_proj = nn.Linear(d_model, d_model)
        self.scale = d_head ** -0.5

    def forward(self, squares: torch.Tensor, cls: torch.Tensor | None = None) -> torch.Tensor:
        b = squares.shape[0]
        feats = squares
        if self.use_cls:
            if cls is None:
                raise ValueError("PointerHead(use_cls=True) requires cls")
            feats = squares + self.cls_proj(cls).unsqueeze(1)   # global conditioner, broadcast
        q = self.q_proj(feats).view(b, 64, self.n_heads, self.d_head)
        k = self.k_proj(feats).view(b, 64, self.n_heads, self.d_head)
        return torch.einsum("bihd,bjhd->bij", q, k) * self.scale   # (B,64,64)

def joint_ce(logits, from_sq, to_sq, legal_mat, label_smoothing: float = 0.0):
    b = logits.shape[0]
    flat = logits.masked_fill(~legal_mat, float("-inf")).view(b, -1)
    target = (from_sq.long() * 64 + to_sq.long())
    return torch.nn.functional.cross_entropy(flat, target, label_smoothing=label_smoothing)
