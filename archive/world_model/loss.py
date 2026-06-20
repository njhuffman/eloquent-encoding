"""world_model loss: patch JEPA/VICReg + masked from-square CE."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F

from jepa2.loss import vicreg_losses
from jepa3.board_square_categories import NUM_SQUARE_CATEGORIES
from jepa3.loss import masked_square_ce


def world_model_loss_forward(
    z_online_global: torch.Tensor,
    patch_online: torch.Tensor,
    patch_hat: torch.Tensor,
    patch_pos: torch.Tensor,
    from_logits: torch.Tensor,
    from_labels: torch.Tensor,
    from_mask: torch.Tensor,
    *,
    jepa_patch_weight: float,
    from_sq_ce_weight: float,
    sq_ce_label_smoothing: float,
    vicreg: dict[str, Any],
    use_amp_cuda: bool,
    piece_recon_logits: torch.Tensor | None = None,
    piece_recon_labels: torch.Tensor | None = None,
    turn_recon_logits: torch.Tensor | None = None,
    turn_recon_labels: torch.Tensor | None = None,
    can_move_logits: torch.Tensor | None = None,
    can_move_labels: torch.Tensor | None = None,
    recon_piece_ce_weight: float = 0.0,
    recon_turn_ce_weight: float = 0.0,
    recon_can_move_ce_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    amp_off = torch.amp.autocast("cuda", enabled=False) if use_amp_cuda else nullcontext()
    with amp_off:
        z_online_global = z_online_global.float()
        patch_online = patch_online.float()
        patch_hat = patch_hat.float()
        patch_pos = patch_pos.float()
        from_logits = from_logits.float()

        d = int(z_online_global.shape[-1])
        if patch_online.ndim != 3 or patch_hat.ndim != 3 or patch_pos.ndim != 3:
            raise ValueError("patch_* must be rank-3 (B, 64, d_model)")
        if patch_online.shape[-1] != d:
            raise ValueError(f"patch_online last dim must match d_model={d}, got {patch_online.shape[-1]}")
        if patch_hat.shape != patch_online.shape or patch_pos.shape != patch_online.shape:
            raise ValueError("patch_hat and patch_pos must match patch_online shape")

        inv_c = float(vicreg["inv_coef"])
        var_c = float(vicreg["var_coef"])
        cov_c = float(vicreg["cov_coef"])
        std_t = float(vicreg["std_target"])
        jpw = float(jepa_patch_weight)
        fw = float(from_sq_ce_weight)
        rpw = float(recon_piece_ce_weight)
        rtw = float(recon_turn_ce_weight)
        rcw = float(recon_can_move_ce_weight)

        b_sz, n_sq, _ = patch_online.shape
        flat_on = patch_online.reshape(b_sz * n_sq, d)
        inv_patch = (patch_hat - patch_pos.detach()).pow(2).mean()
        _, _, _, vic_var_cov_p, m_v_p = vicreg_losses(
            flat_on,
            flat_on,
            flat_on,
            inv_coef=0.0,
            var_coef=var_c,
            cov_coef=cov_c,
            std_target=std_t,
        )
        vic_patch = float(inv_c) * inv_patch + vic_var_cov_p
        m_v_p = {f"patch_{k}": v for k, v in m_v_p.items()}
        m_v_p["patch_vicreg_inv"] = float(inv_patch.detach())
        m_v_p["patch_vicreg_inv_rms"] = float(torch.sqrt(inv_patch.detach().clamp(min=0.0)))

        jepa_p_weighted = jpw * vic_patch

        ce_from, m_from = masked_square_ce(
            from_logits,
            from_labels,
            from_mask,
            label_smoothing=sq_ce_label_smoothing,
        )

        ce_piece_recon = z_online_global.new_zeros(())
        top1_piece_recon = float("nan")
        if rpw > 0.0:
            if piece_recon_logits is None or piece_recon_labels is None:
                raise ValueError("recon_piece_ce_weight > 0 requires piece_recon_logits and piece_recon_labels")
            logits_piece_r = piece_recon_logits.float()
            labels_pr = piece_recon_labels.long()
            if logits_piece_r.ndim != 3 or logits_piece_r.shape[-1] != NUM_SQUARE_CATEGORIES:
                raise ValueError(
                    f"piece_recon_logits must be (B, 64, {NUM_SQUARE_CATEGORIES}), got {tuple(logits_piece_r.shape)}"
                )
            if labels_pr.shape != logits_piece_r.shape[:2]:
                raise ValueError(
                    f"piece_recon_labels must match (B, 64), got {tuple(labels_pr.shape)} vs logits {tuple(logits_piece_r.shape[:2])}"
                )
            ce_piece_recon = F.cross_entropy(
                logits_piece_r.reshape(-1, NUM_SQUARE_CATEGORIES),
                labels_pr.reshape(-1),
            )
            pred = logits_piece_r.argmax(dim=-1)
            top1_piece_recon = float((pred == labels_pr).float().mean().detach() * 100.0)

        ce_turn_recon = z_online_global.new_zeros(())
        top1_turn_recon = float("nan")
        if rtw > 0.0:
            if turn_recon_logits is None or turn_recon_labels is None:
                raise ValueError("recon_turn_ce_weight > 0 requires turn_recon_logits and turn_recon_labels")
            logits_turn_r = turn_recon_logits.float()
            labels_tr = turn_recon_labels.long()
            if logits_turn_r.ndim != 2 or logits_turn_r.shape[-1] != 2:
                raise ValueError(f"turn_recon_logits must be (B, 2), got {tuple(logits_turn_r.shape)}")
            if labels_tr.shape != logits_turn_r.shape[:1]:
                raise ValueError(
                    f"turn_recon_labels must be (B,), got {tuple(labels_tr.shape)} vs logits batch {logits_turn_r.shape[0]}"
                )
            ce_turn_recon = F.cross_entropy(logits_turn_r, labels_tr)
            pred_t = logits_turn_r.argmax(dim=-1)
            top1_turn_recon = float((pred_t == labels_tr).float().mean().detach() * 100.0)

        ce_can_move_recon = z_online_global.new_zeros(())
        top1_can_move_recon = float("nan")
        if rcw > 0.0:
            if can_move_logits is None or can_move_labels is None:
                raise ValueError("recon_can_move_ce_weight > 0 requires can_move_logits and can_move_labels")
            logits_cm = can_move_logits.float()
            labels_cm = can_move_labels.long()
            if logits_cm.ndim != 3 or logits_cm.shape[-1] != 2:
                raise ValueError(f"can_move_logits must be (B, 64, 2), got {tuple(logits_cm.shape)}")
            if labels_cm.shape != logits_cm.shape[:2]:
                raise ValueError(
                    f"can_move_labels must be (B, 64), got {tuple(labels_cm.shape)} vs logits {tuple(logits_cm.shape[:2])}"
                )
            ce_can_move_recon = F.cross_entropy(logits_cm.reshape(-1, 2), labels_cm.reshape(-1))
            pred_cm = logits_cm.argmax(dim=-1)
            top1_can_move_recon = float((pred_cm == labels_cm).float().mean().detach() * 100.0)

        loss = (
            jepa_p_weighted
            + fw * ce_from
            + rpw * ce_piece_recon
            + rtw * ce_turn_recon
            + rcw * ce_can_move_recon
        )

        metrics: dict[str, float] = {
            "loss": float(loss.detach()),
            "jepa_patch_weighted": float(jepa_p_weighted.detach()),
            "jepa_patch_inv_weighted": float((jpw * float(inv_c) * inv_patch).detach()),
            "jepa_patch_varcov_weighted": float((jpw * vic_var_cov_p).detach()),
            "from_sq_ce": float(ce_from.detach()),
            "from_sq_ce_weighted": float((fw * ce_from).detach()),
            "from_sq_top1": m_from["top1"],
            "vicreg_patch_weighted": float(vic_patch.detach()),
            **m_v_p,
        }
        if rpw > 0.0:
            metrics["recon_piece_ce"] = float(ce_piece_recon.detach())
            metrics["recon_piece_ce_weighted"] = float((rpw * ce_piece_recon).detach())
            metrics["recon_piece_top1"] = top1_piece_recon
        if rtw > 0.0:
            metrics["recon_turn_ce"] = float(ce_turn_recon.detach())
            metrics["recon_turn_ce_weighted"] = float((rtw * ce_turn_recon).detach())
            metrics["recon_turn_top1"] = top1_turn_recon
        if rcw > 0.0:
            metrics["recon_can_move_ce"] = float(ce_can_move_recon.detach())
            metrics["recon_can_move_ce_weighted"] = float((rcw * ce_can_move_recon).detach())
            metrics["recon_can_move_top1"] = top1_can_move_recon
    return loss, metrics
