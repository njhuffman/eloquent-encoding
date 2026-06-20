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
    return F.cross_entropy(masked, target.long(), label_smoothing=label_smoothing)


def top1_legal(logits: torch.Tensor, target: torch.Tensor, legal_mask: torch.Tensor) -> float:
    masked = _masked_logits(logits, legal_mask)
    pred = masked.argmax(dim=-1)
    return (pred == target.long()).float().mean().item()
