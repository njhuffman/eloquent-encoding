"""Masked square cross-entropy: softmax restricted to legal squares."""
from __future__ import annotations
import torch
import torch.nn.functional as F

_NEG = float("-inf")


def _masked_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~legal_mask, _NEG)


def masked_square_ce(logits: torch.Tensor, target: torch.Tensor, legal_mask: torch.Tensor,
                     *, label_smoothing: float = 0.0) -> torch.Tensor:
    masked = _masked_logits(logits, legal_mask)
    valid = legal_mask.any(dim=-1)  # (N,) bool — rows with at least one legal square
    if not valid.any():
        # No valid rows: return 0.0 that still participates in autograd
        return (logits * 0.0).sum()
    per_row = F.cross_entropy(masked[valid], target.long()[valid],
                              label_smoothing=label_smoothing, reduction="mean")
    return per_row


def top1_legal(logits: torch.Tensor, target: torch.Tensor, legal_mask: torch.Tensor) -> float:
    masked = _masked_logits(logits, legal_mask)
    pred = masked.argmax(dim=-1)
    return (pred == target.long()).float().mean().item()
