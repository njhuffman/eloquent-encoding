"""Masked square cross-entropy: softmax (and label smoothing) restricted to legal squares."""
from __future__ import annotations
import torch
import torch.nn.functional as F

_NEG = float("-inf")


def _masked_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~legal_mask, _NEG)


def masked_square_ce(logits: torch.Tensor, target: torch.Tensor, legal_mask: torch.Tensor,
                     *, label_smoothing: float = 0.0) -> torch.Tensor:
    """Cross-entropy over legal squares only.

    Illegal squares are excluded from the softmax (set to -inf). Label smoothing, when
    used, spreads mass over the LEGAL squares only — NOT all 64. (Plain
    ``F.cross_entropy(label_smoothing=...)`` would spread mass to the -inf illegal
    classes too, giving ``eps * (-inf) = inf``.) Rows with no legal square are dropped
    (returns a 0.0 that still participates in autograd if every row is empty).
    """
    valid = legal_mask.any(dim=-1)  # (N,) — rows with at least one legal square
    if not valid.any():
        return (logits * 0.0).sum()

    logits = logits[valid]
    legal_mask = legal_mask[valid]
    target = target.long()[valid]

    logp = F.log_softmax(_masked_logits(logits, legal_mask), dim=-1)  # legal: finite, illegal: -inf
    nll = -logp.gather(1, target[:, None]).squeeze(1)                 # (n,)

    if label_smoothing > 0.0:
        n_legal = legal_mask.sum(dim=-1).clamp(min=1)
        logp_legal = torch.where(legal_mask, logp, torch.zeros_like(logp))  # drop -inf before summing
        smooth = -(logp_legal.sum(dim=-1) / n_legal)                        # mean -log p over legal
        per_row = (1.0 - label_smoothing) * nll + label_smoothing * smooth
    else:
        per_row = nll

    return per_row.mean()


def top1_legal(logits: torch.Tensor, target: torch.Tensor, legal_mask: torch.Tensor) -> float:
    masked = _masked_logits(logits, legal_mask)
    pred = masked.argmax(dim=-1)
    return (pred == target.long()).float().mean().item()


def joint_top1(from_logits, from_target, from_mask, to_logits, to_target, to_mask) -> float:
    """Full-move top-1: predicted from-square AND predicted to-square both correct.

    This is the honest autoregressive full-move accuracy (comparable to Maia's ~53%):
    a move is counted correct only when the from-head's top legal square is right AND the
    to-head's top legal square is right. Because correctness requires the from-square to
    match, the (teacher-forced) to-logits are conditioned on the true from-square exactly
    when it equals the predicted one — so no second forward is needed.
    """
    fp = _masked_logits(from_logits, from_mask).argmax(dim=-1)
    tp = _masked_logits(to_logits, to_mask).argmax(dim=-1)
    correct = (fp == from_target.long()) & (tp == to_target.long())
    return correct.float().mean().item()


def wdl_ce(value_logits: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
    """3-class cross-entropy of WDL logits (order loss/draw/win) vs realized result label."""
    return F.cross_entropy(value_logits, result.long())


def wdl_accuracy(value_logits: torch.Tensor, result: torch.Tensor) -> float:
    return (value_logits.argmax(dim=-1) == result.long()).float().mean().item()
