"""Epoch training for jepa3 (JEPA+VICReg + square CE heads + EMA + optional SAM/GSNR).

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
from jepa3.architectures import resolve_config_for_id
from jepa3.checkpoint_paths import stage_checkpoint_path
from jepa3.loss import jepa3_loss_forward


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


def _format_accum_step_log(
    epoch: int,
    completed_opt_steps: int,
    micro_lo: int,
    micro_hi: int,
    *,
    accum_steps: int,
    triples: dict[str, tuple[float, float, float]],
    train_log_mode: str = "compact",
) -> str:
    """Multi-line stderr block (mean over the accumulation window)."""

    def _mean(key: str) -> float:
        return triples.get(key, (float("nan"), float("nan"), float("nan")))[0]

    nan = float("nan")
    l1 = (
        f"epoch={epoch} step={completed_opt_steps} "
        f"micro_batches={micro_lo}-{micro_hi} accum={accum_steps} loss={_mean('loss'):.4f}"
    )
    l2 = (
        f"  vic  cov={_mean('vicreg_cov'):.6f} var={_mean('vicreg_var'):.6f} "
        f"varcov_net={_mean('jepa_varcov_weighted'):.6f}"
    )
    l3 = (
        f"  jepa inv={_mean('vicreg_inv'):.6f} inv_net={_mean('jepa_inv_weighted'):.6f} "
        f"bundle_net={_mean('jepa_weighted'):.6f}"
    )
    l4 = (
        f"  from ce={_mean('from_sq_ce'):.4f} net={_mean('from_sq_ce_weighted'):.4f} "
        f"top1={_mean('from_sq_top1'):.2f}%"
    )
    l5 = (
        f"  to   ce={_mean('to_sq_ce'):.4f} net={_mean('to_sq_ce_weighted'):.4f} "
        f"top1={_mean('to_sq_top1'):.2f}%"
    )
    lines = [l1, l2, l3, l4, l5]
    aux_bits: list[str] = []
    if "aux_board_recon_ce" in triples:
        aux_bits.append(f"br_ce={_mean('aux_board_recon_ce'):.4f}")
        if "aux_board_recon_top1" in triples:
            aux_bits.append(f"br_top1={_mean('aux_board_recon_top1'):.2f}%")
    if "aux_meta_loss" in triples:
        aux_bits.append(f"meta={_mean('aux_meta_loss'):.4f}")
        if "aux_meta_top1" in triples:
            aux_bits.append(f"meta_top1={_mean('aux_meta_top1'):.2f}%")
    if aux_bits:
        lines.append("  aux " + " ".join(aux_bits))
    mode = train_log_mode if train_log_mode in ("compact", "full") else "compact"
    if mode == "full":
        _, ll, lh = triples.get("loss", (nan, nan, nan))
        _, il, ih = triples.get("vicreg_inv", (nan, nan, nan))
        vm, vl, vh = triples.get("vicreg_var", (nan, nan, nan))
        cm, cl, ch = triples.get("vicreg_cov", (nan, nan, nan))
        lines.append(
            f"  ranges loss=[{ll:.4f},{lh:.4f}] inv=[{il:.6f},{ih:.6f}] "
            f"var=[{vl:.6f},{vh:.6f}] cov=[{cl:.6f},{ch:.6f}]"
        )
        if "aux_board_recon_ce" in triples or "aux_meta_loss" in triples:
            pr: list[str] = []
            if "aux_board_recon_ce" in triples:
                _, brl, brh = triples["aux_board_recon_ce"]
                pr.append(f"br_ce=[{brl:.4f},{brh:.4f}]")
            if "aux_board_recon_top1" in triples:
                _, br1l, br1h = triples["aux_board_recon_top1"]
                pr.append(f"br_top1=[{br1l:.2f},{br1h:.2f}]%")
            if "aux_meta_loss" in triples:
                _, ml, mh = triples["aux_meta_loss"]
                pr.append(f"meta=[{ml:.4f},{mh:.4f}]")
            if "aux_meta_top1" in triples:
                _, m1l, m1h = triples["aux_meta_top1"]
                pr.append(f"meta_top1=[{m1l:.2f},{m1h:.2f}]%")
            lines.append("  ranges " + " ".join(pr))
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

    w_br = float(resolved.get("aux_board_recon_weight", 0.0))
    w_meta = float(resolved.get("aux_meta_weight", 0.0))
    run_prefix_aux = hasattr(model, "forward_prefix_aux_losses") and (w_br > 0.0 or w_meta > 0.0)
    aux: dict[str, torch.Tensor] = {}

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
            z_glob_on, z_hat = model.encode_online_with_jepa(
                board_t, fs_dev, ts_dev, from_sq_unk=from_sq_unk
            )
            z_pos = model.encode_target_global(board_post)
            from_logits = model.forward_from_logits(z_glob_on)
            to_logits = model.forward_to_logits(z_glob_on, fs_dev)
            if run_prefix_aux:
                aux = model.forward_prefix_aux_losses(
                    board_t,
                    z_glob_on,
                    compute_board_recon=w_br > 0.0,
                    compute_meta=w_meta > 0.0,
                )
    else:
        z_glob_on, z_hat = model.encode_online_with_jepa(
            board_t, fs_dev, ts_dev, from_sq_unk=from_sq_unk
        )
        z_pos = model.encode_target_global(board_post)
        from_logits = model.forward_from_logits(z_glob_on)
        to_logits = model.forward_to_logits(z_glob_on, fs_dev)
        if run_prefix_aux:
            aux = model.forward_prefix_aux_losses(
                board_t,
                z_glob_on,
                compute_board_recon=w_br > 0.0,
                compute_meta=w_meta > 0.0,
            )

    loss, metrics = jepa3_loss_forward(
        z_glob_on,
        z_hat,
        z_pos,
        from_logits,
        to_logits,
        fs_dev,
        ts_dev,
        from_mask,
        to_mask,
        jepa_weight=float(resolved["jepa_weight"]),
        from_sq_ce_weight=float(resolved["from_sq_ce_weight"]),
        to_sq_ce_weight=float(resolved["to_sq_ce_weight"]),
        sq_ce_label_smoothing=float(resolved["sq_ce_label_smoothing"]),
        vicreg=dict(resolved["vicreg"]),
        use_amp_cuda=bool(use_amp and device.type == "cuda"),
    )
    if run_prefix_aux and aux:
        if w_br > 0.0 and "aux_board_recon" in aux:
            loss = loss + w_br * aux["aux_board_recon"]
            metrics["aux_board_recon_ce"] = float(aux["aux_board_recon"].detach())
            if "aux_board_recon_top1" in aux:
                metrics["aux_board_recon_top1"] = float(aux["aux_board_recon_top1"].detach())
        if w_meta > 0.0 and "aux_meta" in aux:
            loss = loss + w_meta * aux["aux_meta"]
            metrics["aux_meta_loss"] = float(aux["aux_meta"].detach())
            if "aux_meta_top1" in aux:
                metrics["aux_meta_top1"] = float(aux["aux_meta_top1"].detach())
            if "aux_meta_turn_top1" in aux:
                metrics["aux_meta_turn_top1"] = float(aux["aux_meta_turn_top1"].detach())
            if "aux_meta_castle_top1" in aux:
                metrics["aux_meta_castle_top1"] = float(aux["aux_meta_castle_top1"].detach())
            if "aux_meta_ep_top1" in aux:
                metrics["aux_meta_ep_top1"] = float(aux["aux_meta_ep_top1"].detach())
    # ``jepa3_loss_forward`` sets metrics["loss"] before aux; keep it aligned with the scalar we backprop.
    metrics["loss"] = float(loss.detach())
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
    early_stop_joint_top1 = resolved.get("early_stop_joint_top1")

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
                    print(
                        _format_accum_step_log(
                            epoch,
                            completed_opt_steps,
                            micro_lo,
                            bi,
                            accum_steps=accum_steps,
                            triples=triples,
                            train_log_mode=train_log_mode,
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

        if early_stop_joint_top1 is not None:
            tf = inf.get("train_from_sq_top1")
            tt = inf.get("train_to_sq_top1")
            if tf is not None and tt is not None:
                try:
                    joint = (float(tf) / 100.0) * (float(tt) / 100.0)
                except (TypeError, ValueError):
                    joint = float("nan")
                thr = float(early_stop_joint_top1)
                if math.isfinite(joint) and joint > thr:
                    print(
                        f"[early_stop] epoch={epoch} train_joint_top1={joint:.6f} "
                        f"(train_from_top1={float(tf):.2f}% train_to_top1={float(tt):.2f}%) "
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
        alt = repo_root / "jepa3_checkpoints" / name.strip() / "metrics" / path.name
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
        extra={"resolved_training": resolved, "jepa3": True},
    )
    torch.save(payload, out_path)
    return out_path


def init_stage_zero(spec: dict[str, Any], device: torch.device) -> Path:
    from jepa3.architectures import build_model

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
