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
    """Joint cross-entropy over legal (from,to) pairs only.

    Mirrors style_policy.loss.masked_square_ce's label-smoothing handling: illegal pairs
    are excluded from the softmax (-inf) AND from the smoothing mass. Plain
    ``F.cross_entropy(label_smoothing=...)`` would spread eps/4096 mass onto the -inf
    illegal entries too, giving ``eps * -inf = inf``.
    """
    b = logits.shape[0]
    legal_flat = legal_mat.view(b, -1)
    flat = logits.masked_fill(~legal_mat, float("-inf")).view(b, -1)
    target = (from_sq.long() * 64 + to_sq.long())
    logp = torch.nn.functional.log_softmax(flat, dim=-1)
    nll = -logp.gather(1, target[:, None]).squeeze(1)
    if label_smoothing > 0.0:
        n_legal = legal_flat.sum(dim=-1).clamp(min=1)
        logp_legal = torch.where(legal_flat, logp, torch.zeros_like(logp))  # drop -inf before summing
        smooth = -(logp_legal.sum(dim=-1) / n_legal)
        per_row = (1.0 - label_smoothing) * nll + label_smoothing * smooth
    else:
        per_row = nll
    return per_row.mean()
