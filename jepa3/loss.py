"""jepa3 loss: VICReg/JEPA globals + masked from/to square CE."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F

from jepa2.loss import vicreg_losses


def masked_square_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    label_smoothing: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    logits: (B, 64), labels: (B,) int64, mask: (B, 64) float 0/1 legal classes.

    ``torch.nn.functional.cross_entropy`` with ``label_smoothing`` spreads mass over **all**
    classes, which breaks when illegal logits are ``-inf`` (inf/NaN). Match jepa2-style
    masking: smooth only over legal squares, zero ``logp`` on masked slots.
    """
    logits = logits.float()
    mask = mask.float()
    logits_m = logits.masked_fill(mask < 0.5, float("-inf"))
    logp = F.log_softmax(logits_m, dim=-1)
    label_ok = mask.gather(1, labels.long().unsqueeze(1)).squeeze(1) > 0.5
    smooth = float(label_smoothing)

    if smooth > 0.0:
        n_cls = mask.sum(dim=-1).clamp(min=1.0)
        true_dist = torch.zeros_like(logp)
        true_dist.scatter_(1, labels.long().unsqueeze(1), 1.0)
        uniform = mask / n_cls.unsqueeze(-1)
        dist = (1.0 - smooth) * true_dist + smooth * uniform
        dist = dist * mask
        dist = dist / dist.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        logp_f = logp.masked_fill(mask < 0.5, 0.0)
        per_row = -(dist * logp_f).sum(dim=-1)
    else:
        per_row = F.nll_loss(logp, labels, reduction="none")

    denom = label_ok.float().sum().clamp(min=1.0)
    ce = (per_row * label_ok.float()).sum() / denom

    pred = logits_m.argmax(dim=-1)
    if bool(label_ok.any()):
        top1 = ((pred == labels) & label_ok).float().sum() / denom * 100.0
    else:
        top1 = logits.new_zeros(())
    return ce, {"top1": float(top1.detach())}


def jepa3_loss_forward(
    z_online_global: torch.Tensor,
    z_hat: torch.Tensor,
    z_pos_global: torch.Tensor,
    from_logits: torch.Tensor,
    to_logits: torch.Tensor,
    from_labels: torch.Tensor,
    to_labels: torch.Tensor,
    from_mask: torch.Tensor,
    to_mask: torch.Tensor,
    *,
    predictor_prefix_dims: int,
    jepa_weight: float,
    from_sq_ce_weight: float,
    to_sq_ce_weight: float,
    sq_ce_label_smoothing: float,
    vicreg: dict[str, Any],
    use_amp_cuda: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    amp_off = torch.amp.autocast("cuda", enabled=False) if use_amp_cuda else nullcontext()
    with amp_off:
        z_online_global = z_online_global.float()
        z_hat = z_hat.float()
        z_pos_global = z_pos_global.float()
        from_logits = from_logits.float()
        to_logits = to_logits.float()

        p = int(predictor_prefix_dims)
        if z_hat.shape[-1] != p:
            raise ValueError(f"z_hat last dim must equal predictor_prefix_dims={p}, got {z_hat.shape[-1]}")
        if z_pos_global.shape[-1] < p:
            raise ValueError(f"z_pos_global dim {z_pos_global.shape[-1]} < predictor_prefix_dims={p}")

        inv_c = float(vicreg["inv_coef"])
        var_c = float(vicreg["var_coef"])
        cov_c = float(vicreg["cov_coef"])
        std_t = float(vicreg["std_target"])
        jw = float(jepa_weight)
        fw = float(from_sq_ce_weight)
        tw = float(to_sq_ce_weight)
        # Invariance only in the first P dims (predictor output vs target prefix). No gradient on
        # z_pos dimensions beyond P from this term.
        z_pos_p = z_pos_global[:, :p]
        inv_raw = (z_hat - z_pos_p.detach()).pow(2).mean()
        # Variance / covariance on the full online global (all d_model dims).
        _, _, _, vic_var_cov, m_v = vicreg_losses(
            z_online_global,
            z_online_global,
            z_online_global,
            inv_coef=0.0,
            var_coef=var_c,
            cov_coef=cov_c,
            std_target=std_t,
        )
        vic_total = float(inv_c) * inv_raw + vic_var_cov
        m_v = dict(m_v)
        m_v["vicreg_inv"] = float(inv_raw.detach())
        m_v["vicreg_inv_rms"] = float(torch.sqrt(inv_raw.detach().clamp(min=0.0)))

        jepa_inv_weighted = jw * float(inv_c) * inv_raw
        jepa_varcov_weighted = jw * vic_var_cov

        ce_from, m_from = masked_square_ce(
            from_logits,
            from_labels,
            from_mask,
            label_smoothing=sq_ce_label_smoothing,
        )
        ce_to, m_to = masked_square_ce(
            to_logits,
            to_labels,
            to_mask,
            label_smoothing=sq_ce_label_smoothing,
        )

        loss = jw * vic_total + fw * ce_from + tw * ce_to

        metrics: dict[str, float] = {
            "loss": float(loss.detach()),
            "jepa_weighted": float((jw * vic_total).detach()),
            "jepa_inv_weighted": float(jepa_inv_weighted.detach()),
            "jepa_varcov_weighted": float(jepa_varcov_weighted.detach()),
            "from_sq_ce": float(ce_from.detach()),
            "from_sq_ce_weighted": float((fw * ce_from).detach()),
            "to_sq_ce": float(ce_to.detach()),
            "to_sq_ce_weighted": float((tw * ce_to).detach()),
            "from_sq_top1": m_from["top1"],
            "to_sq_top1": m_to["top1"],
            "vicreg_weighted": float(vic_total.detach()),
            **m_v,
        }
    return loss, metrics
