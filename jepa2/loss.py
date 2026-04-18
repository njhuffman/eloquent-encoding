"""jepa2 loss: masked CE (-L2 logits) + MSE(z_hat, z_pos) + VICReg (inv/var/cov)."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F


def vicreg_losses(
    z_online: torch.Tensor,
    z_hat: torch.Tensor,
    z_pos: torch.Tensor,
    *,
    inv_coef: float,
    var_coef: float,
    cov_coef: float,
    std_target: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    z_* : (B, D). Invariance: MSE(z_hat, stopgrad(z_pos)) when inv_coef > 0.
    Variance: mean relu(std_target - per-dim std of z_online over batch).
    Covariance: off-diagonal penalty on centered z_online (VICReg style).
    """
    inv = z_hat.new_zeros(())
    if inv_coef > 0.0:
        inv = (z_hat - z_pos.detach()).pow(2).mean()

    if z_online.shape[0] < 2:
        var_pen = z_online.new_zeros(())
        vicreg_std_mean = z_online.new_zeros(())
        vicreg_std_min = z_online.new_zeros(())
        vicreg_std_max = z_online.new_zeros(())
    else:
        std = z_online.std(dim=0, unbiased=False)
        var_pen = F.relu(float(std_target) - std).mean()
        vicreg_std_mean = std.mean()
        vicreg_std_min = std.min()
        vicreg_std_max = std.max()

    cov_pen = z_online.new_zeros(())
    if cov_coef > 0.0 and z_online.shape[0] >= 2:
        z = z_online - z_online.mean(dim=0, keepdim=True)
        d = z.shape[1]
        if d > 1:
            c = (z.T @ z) / float(z.shape[0] - 1)
            off = c - torch.diag(torch.diag(c))
            cov_pen = (off**2).sum() / float(d * (d - 1))

    total = inv_coef * inv + var_coef * var_pen + cov_coef * cov_pen
    metrics = {
        "vicreg_inv": float(inv.detach()),
        "vicreg_var": float(var_pen.detach()),
        "vicreg_cov": float(cov_pen.detach()),
        "vicreg_std_mean": float(vicreg_std_mean.detach()),
        "vicreg_std_min": float(vicreg_std_min.detach()),
        "vicreg_std_max": float(vicreg_std_max.detach()),
        "vicreg_std_target": float(std_target),
    }
    return inv, var_pen, cov_pen, metrics


def masked_ce_neg_l2_logits(
    z_hat: torch.Tensor,
    z_legals: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    z_hat: (B, D), z_legals: (B, M, D), mask: (B, M) float 0/1, labels: (B,) int in [0, M).
    Logits = -squared L2 per legal.
    """
    # (B, M)
    d = (z_legals - z_hat.unsqueeze(1)).pow(2).sum(dim=-1)
    logits = -d
    logits = logits.masked_fill(mask < 0.5, float("-inf"))
    logp = F.log_softmax(logits, dim=-1)
    if label_smoothing > 0.0:
        n_cls = mask.sum(dim=-1).clamp(min=1.0)
        smooth = float(label_smoothing)
        true_dist = torch.zeros_like(logp)
        true_dist.scatter_(1, labels.unsqueeze(1), 1.0)
        uniform = mask / n_cls.unsqueeze(-1)
        dist = (1.0 - smooth) * true_dist + smooth * uniform
        dist = dist * mask
        dist = dist / dist.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        ce = -(dist * logp).sum(dim=-1).mean()
    else:
        ce = F.nll_loss(logp, labels, reduction="mean")
    pred = logits.argmax(dim=-1)
    valid = mask.sum(dim=-1) > 0
    top1 = ((pred == labels) & valid).float().mean() * 100.0
    # entropy of predicted distribution (for diagnostics)
    p = logp.exp()
    ent = (-(p * logp).masked_fill(mask < 0.5, 0.0)).sum(dim=-1).mean()
    metrics = {
        "ce": float(ce.detach()),
        "top1_acc": float(top1.detach()),
        "softmax_entropy": float(ent.detach()),
        "mean_n_legals_scored": float(mask.sum(dim=-1).mean().detach()),
    }
    return ce, metrics


def mse_hat_to_pos(z_hat: torch.Tensor, z_pos: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    mse = (z_hat - z_pos.detach()).pow(2).mean()
    # RMS over latent dims: same units as z; complements mse_played for "distance" intuition.
    rms = torch.sqrt(mse.detach().clamp(min=0.0))
    return mse, {"mse_played": float(mse.detach()), "mse_hat_pos_rms": float(rms)}


def jepa2_loss_forward(
    z_online: torch.Tensor,
    z_hat: torch.Tensor,
    z_pos: torch.Tensor,
    z_legals: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
    *,
    ce_weight: float,
    mse_played_weight: float,
    ce_label_smoothing: float,
    vicreg: dict[str, Any],
    use_amp_cuda: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    amp_off = torch.amp.autocast("cuda", enabled=False) if use_amp_cuda else nullcontext()
    with amp_off:
        z_online = z_online.float()
        z_hat = z_hat.float()
        z_pos = z_pos.float()
        z_legals = z_legals.float()

        ce, m_ce = masked_ce_neg_l2_logits(
            z_hat, z_legals, mask, labels, label_smoothing=ce_label_smoothing
        )
        mse, m_mse = mse_hat_to_pos(z_hat, z_pos)
        inv_c = float(vicreg.get("inv_coef", 0.0))
        var_c = float(vicreg.get("var_coef", 0.0))
        cov_c = float(vicreg.get("cov_coef", 0.0))
        std_t = float(vicreg.get("std_target", 1.0))
        inv, var_pen, cov_pen, m_v = vicreg_losses(
            z_online,
            z_hat,
            z_pos,
            inv_coef=inv_c,
            var_coef=var_c,
            cov_coef=cov_c,
            std_target=std_t,
        )
        vic_total = inv_c * inv + var_c * var_pen + cov_c * cov_pen

        loss = ce_weight * ce + mse_played_weight * mse + vic_total

        metrics: dict[str, float] = {
            "loss": float(loss.detach()),
            "ce": m_ce["ce"],
            "ce_weighted": float((ce_weight * ce).detach()),
            "mse_played": m_mse["mse_played"],
            "mse_hat_pos_rms": m_mse["mse_hat_pos_rms"],
            "mse_played_weighted": float((mse_played_weight * mse).detach()),
            "top1_acc": m_ce["top1_acc"],
            "softmax_entropy": m_ce["softmax_entropy"],
            "mean_n_legals_scored": m_ce["mean_n_legals_scored"],
            "vicreg_weighted": float(vic_total.detach()),
            **m_v,
        }
    return loss, metrics
