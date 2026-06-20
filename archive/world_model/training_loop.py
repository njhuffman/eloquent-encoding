"""Epoch training for world_model (patch JEPA/VICReg + from-square CE + EMA + optional SAM/GSNR).

Optional global L2 gradient clipping via ``max_gradient_norm`` in the resolved training
config; gradient norms are logged on the same schedule as the multi-line loss log when
``log_gradient_norms`` is true.
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from jepa.checkpoint_utils import build_model_checkpoint
from jepa2.gsnr_probe import flatten_encoder_online_grads, gsnr_metrics_from_grad_vectors
from jepa2.sam import sam_apply_perturbation, sam_build_perturbations, sam_revert_perturbation
from world_model.architectures import resolve_config_for_id
from world_model.checkpoint_paths import stage_checkpoint_path
from world_model.loss import world_model_loss_forward

from jepa3.board_square_categories import square_categories_from_board_tensor

_ELO_BUCKET_EDGES = [1200, 1600, 2000, 2400]
_ELO_BUCKET_LABELS = ["<1.2k", "1.2-1.6k", "1.6-2k", "2-2.4k", "2.4k+"]


def _total_grad_l2_norm(parameters: list[torch.nn.Parameter]) -> float:
    norms: list[torch.Tensor] = []
    for p in parameters:
        if p.grad is None:
            continue
        norms.append(p.grad.detach().float().norm(2))
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms), 2))


def _max_abs_grad(parameters: list[torch.nn.Parameter]) -> float:
    mx: torch.Tensor | None = None
    for p in parameters:
        if p.grad is None:
            continue
        t = p.grad.detach().float().abs().max()
        mx = t if mx is None else torch.maximum(mx, t)
    return float(mx) if mx is not None else 0.0


def _unscale_and_clip_gradients(
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    *,
    use_amp: bool,
    max_gradient_norm: float,
) -> tuple[float, float, float, bool]:
    """
    AMP: unscale grads once, then optional global L2 clip.

    Returns (L2 norm before clip, L2 norm after clip, max |grad| before clip, clipped?).
    """
    params = [p for g in optimizer.param_groups for p in g["params"]]
    if use_amp and scaler is not None:
        scaler.unscale_(optimizer)
    gn_before = _total_grad_l2_norm(params)
    max_abs = _max_abs_grad(params)
    clipped = False
    if max_gradient_norm > 0.0:
        clip_grad_norm_(params, max_norm=max_gradient_norm)
        clipped = gn_before > float(max_gradient_norm) * (1.0 + 1e-6)
    gn_after = _total_grad_l2_norm(params)
    return gn_before, gn_after, max_abs, clipped


def _format_grad_log_line(
    *,
    L2_pre: float,
    L2_post: float,
    max_abs_pre: float,
    clipped: bool,
    max_norm: float,
    train_log_mode: str,
) -> str:
    def _fmt(x: float) -> str:
        if x != x:
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return f"{x:.4e}"

    parts = [
        f"L2_pre={_fmt(L2_pre)}",
        f"L2_post={_fmt(L2_post)}",
        f"max_abs_pre={_fmt(max_abs_pre)}",
        f"clipped={'yes' if clipped else 'no'}",
    ]
    if max_norm > 0.0:
        parts.append(f"max_norm={max_norm:.6g}")
    if L2_pre > 0.0 and math.isfinite(L2_pre) and math.isfinite(L2_post):
        parts.append(f"L2_ratio={L2_post / L2_pre:.4f}")
    if train_log_mode == "full":
        nfin = not (math.isfinite(L2_pre) and math.isfinite(L2_post) and math.isfinite(max_abs_pre))
        parts.append(f"nonfinite={str(nfin)}")
    return "  grad " + " ".join(parts)


def _metric_triples_from_window(ms: list[dict[str, float]]) -> dict[str, tuple[float, float, float]]:
    if not ms:
        return {}
    keys: set[str] = set()
    for m in ms:
        keys.update(m.keys())
    out: dict[str, tuple[float, float, float]] = {}
    for k in keys:
        vals = [float(m[k]) for m in ms if k in m]
        vals = [v for v in vals if v == v]
        if not vals:
            continue
        out[k] = (sum(vals) / len(vals), min(vals), max(vals))
    return out


def _elo_bucket_top1(
    from_logits: torch.Tensor,
    fs: torch.Tensor,
    from_mask: torch.Tensor,
    elo: torch.Tensor,
) -> dict[str, float]:
    """Per-ELO-bucket from-square top1 %, keyed 'elo_top1_b{i}'. NaN when bucket is empty."""
    nan = float("nan")
    logits_m = from_logits.float().masked_fill(from_mask.float() < 0.5, float("-inf"))
    label_ok = from_mask.float().gather(1, fs.long().unsqueeze(1)).squeeze(1) > 0.5
    correct = (logits_m.argmax(-1) == fs.long()) & label_ok
    edges = [float("-inf")] + [float(e) for e in _ELO_BUCKET_EDGES] + [float("inf")]
    out: dict[str, float] = {}
    for i in range(len(_ELO_BUCKET_LABELS)):
        sel = label_ok & (elo >= edges[i]) & (elo < edges[i + 1])
        if sel.any():
            out[f"elo_top1_b{i}"] = float(correct[sel].float().mean().item()) * 100.0
        else:
            out[f"elo_top1_b{i}"] = nan
    return out


def _format_accum_step_log(
    epoch: int,
    completed_opt_steps: int,
    micro_lo: int,
    micro_hi: int,
    *,
    accum_steps: int,
    triples: dict[str, tuple[float, float, float]],
    train_log_mode: str = "compact",
    elo_top1_running: dict[str, float] | None = None,
) -> str:
    """Multi-line stderr block (mean over the accumulation window)."""

    def _mean(key: str) -> float:
        return triples.get(key, (float("nan"), float("nan"), float("nan")))[0]

    nan = float("nan")
    l1 = (
        f"epoch={epoch} step={completed_opt_steps} "
        f"micro_batches={micro_lo}-{micro_hi} accum={accum_steps} loss={_mean('loss'):.4f}"
    )
    l2p = (
        f"  ptc vic  cov={_mean('patch_vicreg_cov'):.6f} var={_mean('patch_vicreg_var'):.6f} "
        f"varcov_net={_mean('jepa_patch_varcov_weighted'):.6f}"
    )
    l3p = (
        f"  ptc jepa inv={_mean('patch_vicreg_inv'):.6f} inv_net={_mean('jepa_patch_inv_weighted'):.6f} "
        f"bundle_net={_mean('jepa_patch_weighted'):.6f}"
    )
    l4 = (
        f"  from ce={_mean('from_sq_ce'):.4f} net={_mean('from_sq_ce_weighted'):.4f} "
        f"top1={_mean('from_sq_top1'):.2f}%"
    )
    lines = [l1, l2p, l3p, l4]
    recon_bits: list[str] = []
    if "recon_piece_ce" in triples:
        recon_bits.append(f"pc_ce={_mean('recon_piece_ce'):.4f}")
        if "recon_piece_top1" in triples:
            recon_bits.append(f"pc_top1={_mean('recon_piece_top1'):.2f}%")
    if "recon_turn_ce" in triples:
        recon_bits.append(f"tn_ce={_mean('recon_turn_ce'):.4f}")
        if "recon_turn_top1" in triples:
            recon_bits.append(f"tn_top1={_mean('recon_turn_top1'):.2f}%")
    if "recon_can_move_ce" in triples:
        recon_bits.append(f"mv_ce={_mean('recon_can_move_ce'):.4f}")
        if "recon_can_move_top1" in triples:
            recon_bits.append(f"mv_top1={_mean('recon_can_move_top1'):.2f}%")
    if recon_bits:
        lines.append("  recon " + " ".join(recon_bits))
    mode = train_log_mode if train_log_mode in ("compact", "full") else "compact"
    if mode == "full":
        _, ll, lh = triples.get("loss", (nan, nan, nan))
        _, il, ih = triples.get("patch_vicreg_inv", (nan, nan, nan))
        vm, vl, vh = triples.get("patch_vicreg_var", (nan, nan, nan))
        cm, cl, ch = triples.get("patch_vicreg_cov", (nan, nan, nan))
        lines.append(
            f"  ranges loss=[{ll:.4f},{lh:.4f}] patch_inv=[{il:.6f},{ih:.6f}] "
            f"patch_var=[{vl:.6f},{vh:.6f}] patch_cov=[{cl:.6f},{ch:.6f}]"
        )
        pr_ranges: list[str] = []
        if "recon_piece_ce" in triples:
            _, pcl, pch = triples["recon_piece_ce"]
            pr_ranges.append(f"pc_ce=[{pcl:.4f},{pch:.4f}]")
        if "recon_piece_top1" in triples:
            _, p1l, p1h = triples["recon_piece_top1"]
            pr_ranges.append(f"pc_top1=[{p1l:.2f},{p1h:.2f}]%")
        if "recon_turn_ce" in triples:
            _, tcl, tch = triples["recon_turn_ce"]
            pr_ranges.append(f"tn_ce=[{tcl:.4f},{tch:.4f}]")
        if "recon_turn_top1" in triples:
            _, t1l, t1h = triples["recon_turn_top1"]
            pr_ranges.append(f"tn_top1=[{t1l:.2f},{t1h:.2f}]%")
        if "recon_can_move_ce" in triples:
            _, mcl, mch = triples["recon_can_move_ce"]
            pr_ranges.append(f"mv_ce=[{mcl:.4f},{mch:.4f}]")
        if "recon_can_move_top1" in triples:
            _, mv1l, mv1h = triples["recon_can_move_top1"]
            pr_ranges.append(f"mv_top1=[{mv1l:.2f},{mv1h:.2f}]%")
        if pr_ranges:
            lines.append("  ranges " + " ".join(pr_ranges))
    if elo_top1_running:
        parts: list[str] = []
        for i, label in enumerate(_ELO_BUCKET_LABELS):
            v = elo_top1_running.get(f"elo_top1_b{i}", float("nan"))
            parts.append(f"{label}={v:.1f}" if v == v else f"{label}=--")
        if parts:
            lines.append("  elo top1/100  " + "  ".join(parts[:-1]) + "  " + parts[-1] + "%")
    return "\n".join(lines)


def _run_encoder_gsnr_probe(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    buffer: list[tuple[Any, ...]],
    *,
    resolved: dict[str, Any],
    device: torch.device,
    use_amp: bool,
    scaler: torch.amp.GradScaler | None,
    accum_steps: int,
    global_step_seed: int,
    epoch: int,
) -> dict[str, float]:
    if not hasattr(model, "encoder_online"):
        return {
            "gsnr_encoder": float("nan"),
            "gsnr_signal": float("nan"),
            "gsnr_noise": float("nan"),
            "gsnr_grad_norm_mean": float("nan"),
        }
    grads_cpu: list[torch.Tensor] = []
    for tup in buffer:
        board_t, board_post, from_mask, to_mask, elo, fs, ts, pr, ep, micro_bi = tup
        board_t = board_t.to(device, non_blocking=True)
        board_post = board_post.to(device, non_blocking=True)
        from_mask = from_mask.to(device, non_blocking=True)
        to_mask = to_mask.to(device, non_blocking=True)
        elo = elo.to(device, non_blocking=True)
        fs = fs.to(device, non_blocking=True)
        ts = ts.to(device, non_blocking=True)
        pr = pr.to(device, non_blocking=True)
        rng = random.Random(global_step_seed + int(ep) * 1_000_000 + int(micro_bi))
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            assert scaler is not None
            with torch.amp.autocast("cuda"):
                loss, _ = _forward_batch(
                    model,
                    board_t,
                    board_post,
                    from_mask,
                    to_mask,
                    elo,
                    fs,
                    ts,
                    pr,
                    resolved,
                    train=True,
                    rng=rng,
                    device=device,
                    use_amp=True,
                )
            scaler.scale(loss / float(accum_steps)).backward()
            scaler.unscale_(optimizer)
        else:
            loss, _ = _forward_batch(
                model,
                board_t,
                board_post,
                from_mask,
                to_mask,
                elo,
                fs,
                ts,
                pr,
                resolved,
                train=True,
                rng=rng,
                device=device,
                use_amp=False,
            )
            (loss / float(accum_steps)).backward()
        grads_cpu.append(flatten_encoder_online_grads(model).cpu().clone())
        optimizer.zero_grad(set_to_none=True)
        if use_amp and scaler is not None:
            scaler.update()
    return gsnr_metrics_from_grad_vectors(grads_cpu)


def _aggregated_encoder_grad_vector(
    model: torch.nn.Module,
    *,
    use_amp: bool,
    scaler: torch.amp.GradScaler | None,
) -> torch.Tensor:
    g = flatten_encoder_online_grads(model)
    if use_amp and scaler is not None:
        scale = float(scaler.get_scale())
        g = g / max(scale, 1e-8)
    return g.detach().cpu().clone()


def _between_step_grad_vectors_for_gsnr(
    step_grad_buf: deque,
    k: int,
) -> list[torch.Tensor] | None:
    if len(step_grad_buf) < 2:
        return None
    lst = list(step_grad_buf)
    if len(lst) >= k:
        return lst[-k:]
    return lst


def _forward_batch(
    model: torch.nn.Module,
    board_t: torch.Tensor,
    board_post: torch.Tensor,
    from_mask: torch.Tensor,
    to_mask: torch.Tensor,
    elo: torch.Tensor,
    fs: torch.Tensor,
    ts: torch.Tensor,
    pr: torch.Tensor,
    resolved: dict[str, Any],
    *,
    train: bool,
    rng: random.Random,
    device: torch.device,
    use_amp: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    del rng
    board_t = board_t.to(device, non_blocking=True)
    board_post = board_post.to(device, non_blocking=True)
    from_mask = from_mask.to(device, non_blocking=True)
    to_mask = to_mask.to(device, non_blocking=True)
    fs_dev = fs.to(device, non_blocking=True, dtype=torch.long)
    ts_dev = ts.to(device, non_blocking=True, dtype=torch.long)
    elo_dev = elo.to(device, non_blocking=True)

    w_piece = float(resolved.get("recon_piece_ce_weight", 0.0))
    w_turn = float(resolved.get("recon_turn_ce_weight", 0.0))
    w_cm = float(resolved.get("recon_can_move_ce_weight", 0.0))
    piece_recon_logits: torch.Tensor | None = None
    piece_recon_labels: torch.Tensor | None = None
    turn_recon_logits: torch.Tensor | None = None
    turn_recon_labels: torch.Tensor | None = None
    can_move_logits: torch.Tensor | None = None
    can_move_labels: torch.Tensor | None = None

    p_unk = float(resolved.get("from_sq_unknown_probability", 0.0))
    from_sq_unk: torch.Tensor | None = None
    if train and p_unk > 0.0:
        if p_unk >= 1.0:
            from_sq_unk = torch.ones(fs_dev.shape[0], device=device, dtype=torch.bool)
        else:
            u = torch.rand(fs_dev.shape[0], device=device, dtype=torch.float32)
            from_sq_unk = u < p_unk

    if use_amp:
        with torch.amp.autocast("cuda"):
            z_glob_on, patch_on, patch_hat = model.encode_online_with_jepa_and_patches(
                board_t, board_post, fs_dev, ts_dev, from_sq_unk=from_sq_unk
            )
            _z_pos, patch_pos = model.encode_target_with_tokens(board_post)
            from_logits = model.forward_from_logits(z_glob_on, patch_on, elo_dev)
            if (w_piece > 0.0 or w_turn > 0.0 or w_cm > 0.0) and hasattr(model, "forward_reconstruction_logits"):
                recon = model.forward_reconstruction_logits(z_glob_on, patch_on)
                if w_piece > 0.0:
                    piece_recon_logits = recon["piece_logits"]
                    piece_recon_labels = square_categories_from_board_tensor(board_t)
                if w_turn > 0.0:
                    turn_recon_logits = recon["turn_logits"]
                    turn_recon_labels = (board_t[:, 0, 0, 12] > 0.5).long()
                if w_cm > 0.0:
                    can_move_logits = recon["can_move_logits"]
                    can_move_labels = (from_mask > 0.5).long()
    else:
        z_glob_on, patch_on, patch_hat = model.encode_online_with_jepa_and_patches(
            board_t, board_post, fs_dev, ts_dev, from_sq_unk=from_sq_unk
        )
        _z_pos, patch_pos = model.encode_target_with_tokens(board_post)
        from_logits = model.forward_from_logits(z_glob_on, patch_on, elo_dev)
        if (w_piece > 0.0 or w_turn > 0.0 or w_cm > 0.0) and hasattr(model, "forward_reconstruction_logits"):
            recon = model.forward_reconstruction_logits(z_glob_on, patch_on)
            if w_piece > 0.0:
                piece_recon_logits = recon["piece_logits"]
                piece_recon_labels = square_categories_from_board_tensor(board_t)
            if w_turn > 0.0:
                turn_recon_logits = recon["turn_logits"]
                turn_recon_labels = (board_t[:, 0, 0, 12] > 0.5).long()
            if w_cm > 0.0:
                can_move_logits = recon["can_move_logits"]
                can_move_labels = (from_mask > 0.5).long()

    loss, metrics = world_model_loss_forward(
        z_glob_on,
        patch_on,
        patch_hat,
        patch_pos,
        from_logits,
        fs_dev,
        from_mask,
        jepa_patch_weight=float(resolved["jepa_patch_weight"]),
        from_sq_ce_weight=float(resolved["from_sq_ce_weight"]),
        sq_ce_label_smoothing=float(resolved["sq_ce_label_smoothing"]),
        vicreg=dict(resolved["vicreg"]),
        use_amp_cuda=bool(use_amp and device.type == "cuda"),
        piece_recon_logits=piece_recon_logits,
        piece_recon_labels=piece_recon_labels,
        turn_recon_logits=turn_recon_logits,
        turn_recon_labels=turn_recon_labels,
        can_move_logits=can_move_logits,
        can_move_labels=can_move_labels,
        recon_piece_ce_weight=w_piece,
        recon_turn_ce_weight=w_turn,
        recon_can_move_ce_weight=w_cm,
    )
    metrics["loss"] = float(loss.detach())
    metrics.update(_elo_bucket_top1(from_logits.detach(), fs_dev, from_mask, elo_dev))
    return loss, metrics


def compute_epoch_metrics_inference(
    model: torch.nn.Module,
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    resolved: dict[str, Any],
    use_amp: bool,
    val_seed: int,
    epoch: int,
    include_train: bool = True,
) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts = {"train": 0, "val": 0}

    def _accum(split: str, m: dict[str, float]) -> None:
        counts[split] += 1
        for k, v in m.items():
            sums[f"{split}_{k}"] = sums.get(f"{split}_{k}", 0.0) + v

    if include_train:
        model.train()
        with torch.no_grad():
            for bi, (board_t, board_post, from_mask, to_mask, elo, fs, ts, pr) in enumerate(train_loader):
                rng = random.Random(val_seed + epoch * 1_000_003 + bi)
                _, m = _forward_batch(
                    model,
                    board_t,
                    board_post,
                    from_mask,
                    to_mask,
                    elo,
                    fs,
                    ts,
                    pr,
                    resolved,
                    train=True,
                    rng=rng,
                    device=device,
                    use_amp=use_amp,
                )
                _accum("train", m)

    model.eval()
    with torch.no_grad():
        for bi, (board_t, board_post, from_mask, to_mask, elo, fs, ts, pr) in enumerate(val_loader):
            rng = random.Random(val_seed + epoch * 1_000_003 + 50_000 + bi)
            _, m = _forward_batch(
                model,
                board_t,
                board_post,
                from_mask,
                to_mask,
                elo,
                fs,
                ts,
                pr,
                resolved,
                train=False,
                rng=rng,
                device=device,
                use_amp=use_amp,
            )
            _accum("val", m)

    out: dict[str, float] = {}
    for split in ("train", "val"):
        n = max(counts[split], 1)
        for k, total in sums.items():
            if k.startswith(f"{split}_"):
                out[k] = total / n
    return out


def run_training_epochs(
    model: torch.nn.Module,
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    resolved: dict[str, Any],
    metrics_run_meta: dict[str, Any] | None = None,
    global_step_seed: int = 0,
) -> tuple[float, int, dict[str, float], float, dict[str, float], int, bool]:
    tr = resolved["train"]
    epochs = int(tr["epochs"])
    lr = float(tr["learning_rate"])
    wd = float(tr["weight_decay"])
    use_amp = bool(resolved.get("use_amp", True)) and device.type == "cuda"
    ema_momentum = float(resolved["ema_momentum"])
    log_interval = int(resolved.get("log_interval", 100))
    accum_steps = max(int(tr.get("gradient_accumulation_steps", 1)), 1)
    train_log_mode = str(resolved.get("train_log_mode", "compact"))
    max_gradient_norm = float(resolved["max_gradient_norm"])
    log_gradient_norms = bool(resolved["log_gradient_norms"])
    early_stop_from_sq_top1 = resolved.get("early_stop_from_sq_top1")

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    best_val = float("inf")
    best_ep = 0
    last_inf: dict[str, float] = {}
    last_train_epoch_loss = 0.0
    last_avg_train: dict[str, float] = {}

    gsnr_k = int(resolved.get("gsnr_probe_k", 8))
    gsnr_every = int(resolved.get("gsnr_probe_every_opt_steps", 0))
    sam_rho = float(resolved.get("sam_rho", 0.0))
    need_replay_buffer = gsnr_every > 0 or sam_rho > 0.0
    replay_batch_window: list[tuple[Any, ...]] = []
    step_grad_buf: deque | None = deque(maxlen=gsnr_k) if gsnr_every > 0 else None
    elo_top1_100: deque[dict[str, float]] = deque(maxlen=100)
    last_gsnr_within: float | None = None
    last_gsnr_between: float | None = None
    epochs_ran = 0
    early_stopped = False

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        sum_metrics: dict[str, float] = {}
        gsnr_within_epoch: list[float] = []
        gsnr_between_epoch: list[float] = []

        n_micro_epoch = len(train_loader)
        micro_in_accum = 0
        completed_opt_steps = 0
        accum_ms: list[dict[str, float]] = []
        micro_lo = 0

        for bi, (board_t, board_post, from_mask, to_mask, elo, fs, ts, pr) in enumerate(train_loader):
            rng = random.Random(global_step_seed + epoch * 1_000_000 + bi)
            if micro_in_accum == 0:
                optimizer.zero_grad(set_to_none=True)
                accum_ms.clear()
                micro_lo = bi
                if need_replay_buffer:
                    replay_batch_window.clear()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    loss, m = _forward_batch(
                        model,
                        board_t,
                        board_post,
                        from_mask,
                        to_mask,
                        elo,
                        fs,
                        ts,
                        pr,
                        resolved,
                        train=True,
                        rng=rng,
                        device=device,
                        use_amp=use_amp,
                    )
                scaler.scale(loss / float(accum_steps)).backward()
            else:
                loss, m = _forward_batch(
                    model,
                    board_t,
                    board_post,
                    from_mask,
                    to_mask,
                    elo,
                    fs,
                    ts,
                    pr,
                    resolved,
                    train=True,
                    rng=rng,
                    device=device,
                    use_amp=False,
                )
                (loss / float(accum_steps)).backward()

            train_loss += float(loss.detach())
            n_batches += 1
            for k, v in m.items():
                sum_metrics[k] = sum_metrics.get(k, 0.0) + v
            accum_ms.append(dict(m))

            if need_replay_buffer:
                replay_batch_window.append(
                    (
                        board_t.detach().cpu(),
                        board_post.detach().cpu(),
                        from_mask.detach().cpu(),
                        to_mask.detach().cpu(),
                        elo.detach().cpu(),
                        fs.detach().cpu(),
                        ts.detach().cpu(),
                        pr.detach().cpu(),
                        epoch,
                        bi,
                    )
                )

            micro_in_accum += 1
            is_last_micro = bi == n_micro_epoch - 1
            should_step = micro_in_accum == accum_steps or (is_last_micro and 0 < micro_in_accum < accum_steps)

            if should_step:
                replay_batches = list(replay_batch_window) if need_replay_buffer else []

                if gsnr_every > 0 and step_grad_buf is not None and hasattr(model, "encoder_online"):
                    step_grad_buf.append(
                        _aggregated_encoder_grad_vector(model, use_amp=use_amp, scaler=scaler)
                    )

                if sam_rho > 0.0:
                    trainables = list(model.trainable_parameters())
                    perturbations = sam_build_perturbations(trainables, sam_rho)
                    if perturbations:
                        sam_apply_perturbation(perturbations)
                        optimizer.zero_grad(set_to_none=True)
                        for tup in replay_batches:
                            board_t2, board_post2, from_m2, to_m2, elo2, fs2, ts2, pr2, ep, micro_bi = tup
                            board_t2 = board_t2.to(device, non_blocking=True)
                            board_post2 = board_post2.to(device, non_blocking=True)
                            from_m2 = from_m2.to(device, non_blocking=True)
                            to_m2 = to_m2.to(device, non_blocking=True)
                            elo2 = elo2.to(device, non_blocking=True)
                            fs2 = fs2.to(device, non_blocking=True)
                            ts2 = ts2.to(device, non_blocking=True)
                            pr2 = pr2.to(device, non_blocking=True)
                            rng = random.Random(global_step_seed + int(ep) * 1_000_000 + int(micro_bi))
                            if use_amp:
                                assert scaler is not None
                                with torch.amp.autocast("cuda"):
                                    loss2, _ = _forward_batch(
                                        model,
                                        board_t2,
                                        board_post2,
                                        from_m2,
                                        to_m2,
                                        elo2,
                                        fs2,
                                        ts2,
                                        pr2,
                                        resolved,
                                        train=True,
                                        rng=rng,
                                        device=device,
                                        use_amp=True,
                                    )
                                scaler.scale(loss2 / float(accum_steps)).backward()
                            else:
                                loss2, _ = _forward_batch(
                                    model,
                                    board_t2,
                                    board_post2,
                                    from_m2,
                                    to_m2,
                                    elo2,
                                    fs2,
                                    ts2,
                                    pr2,
                                    resolved,
                                    train=True,
                                    rng=rng,
                                    device=device,
                                    use_amp=False,
                                )
                                (loss2 / float(accum_steps)).backward()
                        sam_revert_perturbation(perturbations)

                gn_before, gn_after, max_abs, grad_clipped = _unscale_and_clip_gradients(
                    optimizer,
                    scaler,
                    use_amp=use_amp,
                    max_gradient_norm=max_gradient_norm,
                )

                if use_amp:
                    assert scaler is not None
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                model.ema_update_target(ema_momentum)
                completed_opt_steps += 1
                triples = _metric_triples_from_window(accum_ms)
                elo_top1_100.append({k: triples[k][0] for k in triples if k.startswith("elo_top1_b")})

                if gsnr_every > 0 and completed_opt_steps % gsnr_every == 0:

                    def _fmt_gsnr(x: float) -> str:
                        return f"{x:.6e}" if x == x else "nan"

                    nan = float("nan")
                    within_g = nan
                    between_g = nan
                    ws = wn = wgmn = nan
                    bs = bn = bgmn = nan

                    if len(replay_batches) >= 2:
                        w_out = _run_encoder_gsnr_probe(
                            model,
                            optimizer,
                            replay_batches,
                            resolved=resolved,
                            device=device,
                            use_amp=use_amp,
                            scaler=scaler,
                            accum_steps=accum_steps,
                            global_step_seed=global_step_seed,
                            epoch=epoch,
                        )
                        within_g = float(w_out["gsnr_encoder"])
                        ws = float(w_out["gsnr_signal"])
                        wn = float(w_out["gsnr_noise"])
                        wgmn = float(w_out["gsnr_grad_norm_mean"])
                        if math.isfinite(within_g):
                            last_gsnr_within = within_g
                            gsnr_within_epoch.append(within_g)

                    between_vecs = (
                        _between_step_grad_vectors_for_gsnr(step_grad_buf, gsnr_k)
                        if step_grad_buf is not None
                        else None
                    )
                    if between_vecs is not None:
                        b_out = gsnr_metrics_from_grad_vectors(between_vecs)
                        between_g = float(b_out["gsnr_encoder"])
                        bs = float(b_out["gsnr_signal"])
                        bn = float(b_out["gsnr_noise"])
                        bgmn = float(b_out["gsnr_grad_norm_mean"])
                        if math.isfinite(between_g):
                            last_gsnr_between = between_g
                            gsnr_between_epoch.append(between_g)

                    if train_log_mode == "full":
                        print(
                            f"[gsnr] epoch={epoch} opt_step={completed_opt_steps} "
                            f"within_update={_fmt_gsnr(within_g)} signal_w={_fmt_gsnr(ws)} noise_w={_fmt_gsnr(wn)} "
                            f"grad_norm_mean_w={_fmt_gsnr(wgmn)} "
                            f"between_updates={_fmt_gsnr(between_g)} signal_b={_fmt_gsnr(bs)} noise_b={_fmt_gsnr(bn)} "
                            f"grad_norm_mean_b={_fmt_gsnr(bgmn)}",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"[gsnr] e{epoch} step{completed_opt_steps} "
                            f"within={_fmt_gsnr(within_g)} between={_fmt_gsnr(between_g)}",
                            file=sys.stderr,
                        )

                if log_interval and (completed_opt_steps - 1) % log_interval == 0:
                    elo_running: dict[str, float] = {}
                    for _bi in range(len(_ELO_BUCKET_LABELS)):
                        _key = f"elo_top1_b{_bi}"
                        _vals = [d[_key] for d in elo_top1_100 if _key in d and d[_key] == d[_key]]
                        elo_running[_key] = sum(_vals) / len(_vals) if _vals else float("nan")
                    print(
                        _format_accum_step_log(
                            epoch,
                            completed_opt_steps,
                            micro_lo,
                            bi,
                            accum_steps=accum_steps,
                            triples=triples,
                            train_log_mode=train_log_mode,
                            elo_top1_running=elo_running,
                        ),
                        file=sys.stderr,
                    )
                    if log_gradient_norms:
                        print(
                            _format_grad_log_line(
                                L2_pre=gn_before,
                                L2_post=gn_after,
                                max_abs_pre=max_abs,
                                clipped=grad_clipped,
                                max_norm=max_gradient_norm,
                                train_log_mode=train_log_mode,
                            ),
                            file=sys.stderr,
                        )
                micro_in_accum = 0

        train_loss /= max(n_batches, 1)
        avg_train = {k: sum_metrics[k] / max(n_batches, 1) for k in sum_metrics}
        if gsnr_within_epoch:
            avg_train["gsnr_within_update"] = sum(gsnr_within_epoch) / len(gsnr_within_epoch)
        if gsnr_between_epoch:
            avg_train["gsnr_between_updates"] = sum(gsnr_between_epoch) / len(gsnr_between_epoch)

        inf = compute_epoch_metrics_inference(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            resolved=resolved,
            use_amp=use_amp,
            val_seed=int(resolved.get("val_legal_seed", 42)),
            epoch=epoch,
            include_train=False,
        )
        for k, v in avg_train.items():
            inf[f"train_{k}"] = v
        val_loss = inf.get("val_loss", train_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_ep = epoch

        scheduler.step()
        last_inf = inf
        last_train_epoch_loss = train_loss
        last_avg_train = avg_train
        epochs_ran = epoch

        if early_stop_from_sq_top1 is not None:
            tf = inf.get("train_from_sq_top1")
            if tf is not None:
                try:
                    from_frac = float(tf) / 100.0
                except (TypeError, ValueError):
                    from_frac = float("nan")
                thr = float(early_stop_from_sq_top1)
                if math.isfinite(from_frac) and from_frac > thr:
                    print(
                        f"[early_stop] epoch={epoch} train_from_sq_top1_frac={from_frac:.6f} "
                        f"(train_from_top1={float(tf):.2f}%) "
                        f"> threshold={thr:.6f}",
                        file=sys.stderr,
                    )
                    early_stopped = True
                    break

    return best_val, best_ep, last_inf, last_train_epoch_loss, last_avg_train, epochs_ran, early_stopped


def write_stage_metrics_json(path: Path, record: dict[str, Any]) -> Path:
    text = json.dumps(record, indent=2, default=str)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(text, encoding="utf-8")
        return path
    except PermissionError:
        name = record.get("model")
        if not isinstance(name, str) or not name.strip():
            raise
        repo_root = Path(__file__).resolve().parents[1]
        alt = repo_root / "world_model_checkpoints" / name.strip() / "metrics" / path.name
        alt.parent.mkdir(parents=True, exist_ok=True)
        alt.write_text(text, encoding="utf-8")
        print(
            f"Warning: could not write metrics to {path}; wrote to {alt} (fix ownership or checkpoint_dir).",
            file=sys.stderr,
        )
        return alt


def save_stage_checkpoint(
    *,
    model: torch.nn.Module,
    spec: dict[str, Any],
    stage: int,
    resolved: dict[str, Any],
    train_meta: dict[str, Any],
    best_val: float,
    best_ep: int,
    epochs_ran: int,
) -> Path:
    name = spec["name"]
    ckpt_dir = Path(spec["checkpoint_dir"])
    arch_id = spec["architecture"]["id"]
    arch_cfg = spec["architecture"].get("config") or {}
    resolved_arch = resolve_config_for_id(arch_id, arch_cfg)
    out_path = stage_checkpoint_path(ckpt_dir, name, stage)
    spec_snap = {
        "name": name,
        "stage": stage,
        "defaults": spec["defaults"],
        "stages": spec["stages"],
        "architecture": spec["architecture"],
        "resolved_training": resolved,
    }
    payload = build_model_checkpoint(
        model,
        architecture_id=arch_id,
        architecture_config=resolved_arch,
        train_meta=train_meta,
        train_hparams={
            "stage": stage,
            "best_val_loss": best_val,
            "best_epoch": best_ep,
            "epochs_ran": epochs_ran,
        },
        training_spec=spec_snap,
        extra={"resolved_training": resolved, "world_model": True},
    )
    torch.save(payload, out_path)
    return out_path


def init_stage_zero(spec: dict[str, Any], device: torch.device) -> Path:
    from world_model.architectures import build_model

    name = spec["name"]
    ckpt_dir = Path(spec["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    arch_id = spec["architecture"]["id"]
    arch_cfg = spec["architecture"].get("config") or {}
    model = build_model(arch_id, arch_cfg).to(device)
    model.init_target_from_online()
    resolved: dict[str, Any] = dict(spec.get("defaults") or {})
    out = save_stage_checkpoint(
        model=model,
        spec=spec,
        stage=0,
        resolved=resolved,
        train_meta={"init_only": True, "storage": "streaming"},
        best_val=float("nan"),
        best_ep=0,
        epochs_ran=0,
    )
    return out
