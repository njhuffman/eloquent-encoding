"""jepa2 loss: masked CE (-L2 logits) + VICReg (inv / var / cov on z_online)."""

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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:  # inv, var, cov, vic_total, metrics
    """
    z_* : (B, D). Invariance: mean squared (z_hat - stopgrad(z_pos)) — scaled by inv_coef in the caller.
    Variance: mean squared deviation of per-dim batch std of z_online from std_target (symmetric: below and above target).
    Covariance: off-diagonal penalty on centered z_online (VICReg style).
    """
    inv_raw = (z_hat - z_pos.detach()).pow(2).mean()

    if z_online.shape[0] < 2:
        var_pen = z_online.new_zeros(())
        vicreg_std_mean = z_online.new_zeros(())
        vicreg_std_min = z_online.new_zeros(())
        vicreg_std_max = z_online.new_zeros(())
    else:
        std = z_online.std(dim=0, unbiased=False)
        t = z_online.new_tensor(float(std_target), dtype=std.dtype, device=std.device)
        var_pen = (std - t).pow(2).mean()
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

    total = float(inv_coef) * inv_raw + float(var_coef) * var_pen + float(cov_coef) * cov_pen
    metrics = {
        "vicreg_inv": float(inv_raw.detach()),
        "vicreg_inv_rms": float(torch.sqrt(inv_raw.detach().clamp(min=0.0))),
        "vicreg_var": float(var_pen.detach()),
        "vicreg_cov": float(cov_pen.detach()),
        "vicreg_std_mean": float(vicreg_std_mean.detach()),
        "vicreg_std_min": float(vicreg_std_min.detach()),
        "vicreg_std_max": float(vicreg_std_max.detach()),
        "vicreg_std_target": float(std_target),
    }
    return inv_raw, var_pen, cov_pen, total, metrics


def succ_vicreg_losses(
    z_online_legals: torch.Tensor,
    mask: torch.Tensor,
    *,
    std_target: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    VICReg-style variance / covariance across **legal successor** slots (dim ``M``) per row.

    ``z_online_legals``: (B, M, D) from the online encoder on ``succ``; ``mask`` (B, M) marks
    real legals. Rows with fewer than 2 legals are skipped.     Returns raw ``var_raw`` / ``cov_raw`` tensors (mean penalty per row over qualifying rows);
    the caller forms ``succ_var_coef * var_raw + succ_cov_coef * cov_raw`` for the loss.
    """
    z = z_online_legals
    m = mask.float()
    count = m.sum(dim=1)
    valid = count >= 2.0
    t = z.new_tensor(float(std_target), dtype=z.dtype, device=z.device)
    bsz, _m, d_model = z.shape

    if not bool(valid.any()):
        zero = z.sum() * 0.0
        meta = {
            "succ_vicreg_var": 0.0,
            "succ_vicreg_cov": 0.0,
            "succ_vicreg_weighted": 0.0,
            "succ_vicreg_std_mean": 0.0,
        }
        return zero, zero, meta

    count_safe = count.clamp(min=1e-8)
    mean = (z * m.unsqueeze(-1)).sum(dim=1) / count_safe.unsqueeze(-1)
    zc = z - mean.unsqueeze(1)
    var_md = (zc.pow(2) * m.unsqueeze(-1)).sum(dim=1) / count_safe.unsqueeze(-1)
    std_md = var_md.sqrt().clamp(min=0.0)
    var_pen_row = (std_md - t).pow(2).mean(dim=1)
    var_pen_row = torch.where(valid, var_pen_row, torch.zeros_like(var_pen_row))
    n_qual = valid.float().sum().clamp(min=1.0)
    var_raw = var_pen_row.sum() / n_qual

    std_mean_diag = (std_md.mean(dim=1) * valid.float()).sum() / n_qual

    cov_raw = z.sum() * 0.0
    if d_model > 1:
        cov_acc = z.new_zeros(())
        cov_n = 0
        for b in range(bsz):
            if not bool(valid[b].item()):
                continue
            idx = m[b] > 0.5
            zb = z[b, idx]
            n = int(zb.shape[0])
            if n < 2:
                continue
            zr = zb - zb.mean(dim=0, keepdim=True)
            c = (zr.T @ zr) / float(n - 1)
            off = c - torch.diag(torch.diag(c))
            cov_pen = (off**2).sum() / float(d_model * (d_model - 1))
            cov_acc = cov_acc + cov_pen
            cov_n += 1
        if cov_n > 0:
            cov_raw = cov_acc / float(cov_n)

    meta = {
        "succ_vicreg_var": float(var_raw.detach()),
        "succ_vicreg_cov": float(cov_raw.detach()),
        "succ_vicreg_std_mean": float(std_mean_diag.detach()),
    }
    return var_raw, cov_raw, meta


def masked_ce_neg_l2_logits(
    z_hat: torch.Tensor,
    z_legals: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    z_hat: (B, D), z_legals: (B, M, D), mask: (B, M) float 0/1, labels: (B,) int in [0, M).
    Logits = -squared L2 per legal, then divided by ``temperature`` before softmax.
    ``ce_logit_true_mean`` is the batch mean of the true-move logit (same scaled logits as softmax).
    """
    # (B, M)
    d = (z_legals - z_hat.unsqueeze(1)).pow(2).sum(dim=-1)
    logits = -d
    logits = logits.masked_fill(mask < 0.5, float("-inf"))
    t = max(float(temperature), 1e-8)
    logits = logits / z_hat.new_tensor(t, dtype=logits.dtype, device=logits.device)
    legals = mask > 0.5
    if bool(legals.any()):
        lv = logits[legals]
        ce_logit_min = float(lv.min().detach())
        ce_logit_max = float(lv.max().detach())
    else:
        ce_logit_min = float("nan")
        ce_logit_max = float("nan")
    valid = mask.sum(dim=-1) > 0
    if bool(valid.any()):
        true_l = logits.gather(1, labels.long().view(-1, 1)).squeeze(1)
        ce_logit_true_mean = float(true_l[valid].mean().detach())
    else:
        ce_logit_true_mean = float("nan")
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
        # logp is -inf on masked slots; dist is 0 there but 0 * (-inf) -> NaN in torch.
        logp_f = logp.masked_fill(mask < 0.5, 0.0)
        ce = -(dist * logp_f).sum(dim=-1).mean()
    else:
        ce = F.nll_loss(logp, labels, reduction="mean")
    pred = logits.argmax(dim=-1)
    top1 = ((pred == labels) & valid).float().mean() * 100.0
    # entropy of predicted distribution (for diagnostics)
    p = logp.exp()
    ent = (-(p * logp).masked_fill(mask < 0.5, 0.0)).sum(dim=-1).mean()
    metrics = {
        "ce": float(ce.detach()),
        "top1_acc": float(top1.detach()),
        "softmax_entropy": float(ent.detach()),
        "mean_n_legals_scored": float(mask.sum(dim=-1).mean().detach()),
        "ce_logit_min": ce_logit_min,
        "ce_logit_max": ce_logit_max,
        "ce_logit_true_mean": ce_logit_true_mean,
    }
    return ce, metrics


def jepa2_loss_forward(
    z_online: torch.Tensor,
    z_hat: torch.Tensor,
    z_pos: torch.Tensor,
    z_legals: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
    *,
    ce_weight: float,
    ce_label_smoothing: float,
    ce_temperature: float,
    vicreg: dict[str, Any],
    use_amp_cuda: bool,
    z_online_legals: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    amp_off = torch.amp.autocast("cuda", enabled=False) if use_amp_cuda else nullcontext()
    with amp_off:
        z_online = z_online.float()
        z_hat = z_hat.float()
        z_pos = z_pos.float()
        z_legals = z_legals.float()

        ce, m_ce = masked_ce_neg_l2_logits(
            z_hat,
            z_legals,
            mask,
            labels,
            label_smoothing=ce_label_smoothing,
            temperature=ce_temperature,
        )
        inv_c = float(vicreg.get("inv_coef", 0.0))
        var_c = float(vicreg.get("var_coef", 0.0))
        cov_c = float(vicreg.get("cov_coef", 0.0))
        std_t = float(vicreg.get("std_target", 1.0))
        inv_raw, var_pen, cov_pen, vic_total, m_v = vicreg_losses(
            z_online,
            z_hat,
            z_pos,
            inv_coef=inv_c,
            var_coef=var_c,
            cov_coef=cov_c,
            std_target=std_t,
        )

        succ_vc = float(vicreg.get("succ_var_coef", 0.0))
        succ_cc = float(vicreg.get("succ_cov_coef", 0.0))
        succ_std_t = float(vicreg.get("succ_std_target", vicreg.get("std_target", 1.0)))
        if succ_vc > 0.0 or succ_cc > 0.0:
            if z_online_legals is None:
                raise ValueError("z_online_legals required when succ_var_coef or succ_cov_coef > 0")
            z_succ = z_online_legals.float()
            var_raw_s, cov_raw_s, m_succ = succ_vicreg_losses(
                z_succ,
                mask,
                std_target=succ_std_t,
            )
            succ_w = succ_vc * var_raw_s + succ_cc * cov_raw_s
            m_succ = dict(m_succ)
            m_succ["succ_vicreg_weighted"] = float(succ_w.detach())
        else:
            m_succ = {
                "succ_vicreg_var": 0.0,
                "succ_vicreg_cov": 0.0,
                "succ_vicreg_weighted": 0.0,
                "succ_vicreg_std_mean": 0.0,
            }
            succ_w = z_online.new_zeros(())

        loss = ce_weight * ce + vic_total + succ_w

        metrics: dict[str, float] = {
            "loss": float(loss.detach()),
            "ce": m_ce["ce"],
            "ce_weighted": float((ce_weight * ce).detach()),
            "ce_temperature": float(ce_temperature),
            "ce_logit_min": m_ce["ce_logit_min"],
            "ce_logit_max": m_ce["ce_logit_max"],
            "ce_logit_true_mean": m_ce["ce_logit_true_mean"],
            "top1_acc": m_ce["top1_acc"],
            "softmax_entropy": m_ce["softmax_entropy"],
            "mean_n_legals_scored": m_ce["mean_n_legals_scored"],
            "vicreg_weighted": float(vic_total.detach()),
            **m_succ,
            **m_v,
        }
    return loss, metrics
