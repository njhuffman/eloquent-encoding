"""Staged rfp training loop (AdamW, optional AMP, grad accumulation, cosine LR)."""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from jepa3.loss import masked_square_ce

from rfp.model import ResidualFromPredictor


def _forward_rfp_batch(
    model: ResidualFromPredictor,
    delta_z: torch.Tensor,
    z_curr: torch.Tensor,
    hist_mask: torch.Tensor,
    elo_bucket: torch.Tensor,
    from_m: torch.Tensor,
    fs: torch.Tensor,
    *,
    label_smoothing: float,
    elo_null_prob: float,
    train: bool,
    device: torch.device,
    use_amp: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    delta_z = delta_z.to(device, non_blocking=True)
    z_curr = z_curr.to(device, non_blocking=True)
    hist_mask = hist_mask.to(device, non_blocking=True)
    elo_bucket = elo_bucket.to(device, non_blocking=True)
    from_m = from_m.to(device, non_blocking=True)
    fs = fs.to(device, non_blocking=True)

    ctx = torch.set_grad_enabled(train)
    amp_on = bool(use_amp and device.type == "cuda")
    with ctx:
        if amp_on:
            with torch.amp.autocast("cuda"):
                logits, _gfp = model(
                    delta_z,
                    z_curr,
                    hist_mask,
                    elo_bucket,
                    elo_null_prob=float(elo_null_prob) if train else 0.0,
                    train=train,
                )
                loss, m = masked_square_ce(
                    logits, fs, from_m, label_smoothing=float(label_smoothing)
                )
        else:
            logits, _gfp = model(
                delta_z,
                z_curr,
                hist_mask,
                elo_bucket,
                elo_null_prob=float(elo_null_prob) if train else 0.0,
                train=train,
            )
            loss, m = masked_square_ce(
                logits, fs, from_m, label_smoothing=float(label_smoothing)
            )
    return loss, m


def run_rfp_training_epochs(
    model: ResidualFromPredictor,
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    resolved: dict[str, Any],
    metrics_run_meta: dict[str, Any] | None = None,
) -> tuple[float, int, dict[str, float], float, dict[str, float], int, bool]:
    tr = resolved["train"]
    epochs = int(tr["epochs"])
    lr = float(tr["learning_rate"])
    wd = float(tr["weight_decay"])
    label_smoothing = float(resolved["sq_ce_label_smoothing"])
    elo_null_prob = float(resolved["elo_null_prob"])
    use_amp = bool(resolved.get("use_amp", True)) and device.type == "cuda"
    accum_steps = max(int(tr.get("gradient_accumulation_steps", 1)), 1)
    log_interval = int(resolved.get("log_interval", 100))
    max_gradient_norm = float(resolved["max_gradient_norm"])
    early_thr = resolved.get("early_stop_train_top1")
    top1_roll_window = int(resolved.get("train_log_top1_window", 100))
    top1_ema_beta = float(resolved.get("train_log_top1_ema_beta", 0.99))

    optimizer = AdamW(model.trainable_parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_val = float("inf")
    best_ep = 0
    last_val: dict[str, float] = {}
    last_train_loss = 0.0
    last_avg_train: dict[str, float] = {}
    epochs_ran = 0
    early_stopped = False

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        sum_top1 = 0.0
        top1_recent: deque[float] = deque(maxlen=max(top1_roll_window, 1))
        top1_ema: float | None = None
        n_micro = len(train_loader)
        if n_micro == 0:
            raise RuntimeError(
                "train_loader is empty (batch_size too large vs dataset, or drop_last removed all batches)"
            )
        micro_in_accum = 0

        for bi, (delta_z, z_curr, hist_mask, from_m, fs, elo_bucket) in enumerate(train_loader):
            if micro_in_accum == 0:
                optimizer.zero_grad(set_to_none=True)

            if use_amp:
                assert scaler is not None
                with torch.amp.autocast("cuda"):
                    loss, m = _forward_rfp_batch(
                        model,
                        delta_z,
                        z_curr,
                        hist_mask,
                        elo_bucket,
                        from_m,
                        fs,
                        label_smoothing=label_smoothing,
                        elo_null_prob=elo_null_prob,
                        train=True,
                        device=device,
                        use_amp=True,
                    )
                scaler.scale(loss / float(accum_steps)).backward()
            else:
                loss, m = _forward_rfp_batch(
                    model,
                    delta_z,
                    z_curr,
                    hist_mask,
                    elo_bucket,
                    from_m,
                    fs,
                    label_smoothing=label_smoothing,
                    elo_null_prob=elo_null_prob,
                    train=True,
                    device=device,
                    use_amp=False,
                )
                (loss / float(accum_steps)).backward()

            train_loss += float(loss.detach())
            n_batches += 1
            t1 = float(m["top1"])
            sum_top1 += t1
            top1_recent.append(t1)
            if top1_ema is None:
                top1_ema = t1
            else:
                top1_ema = top1_ema_beta * top1_ema + (1.0 - top1_ema_beta) * t1

            micro_in_accum += 1
            is_last_micro = bi == n_micro - 1
            should_step = micro_in_accum == accum_steps or (
                is_last_micro and 0 < micro_in_accum < accum_steps
            )

            if should_step:
                if max_gradient_norm > 0.0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.trainable_parameters(), max_gradient_norm
                    )
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                micro_in_accum = 0

            if log_interval > 0 and (bi + 1) % log_interval == 0:
                meta = metrics_run_meta or {}
                top1_batch = float(m["top1"])
                top1_epoch_mean = sum_top1 / float(n_batches)
                roll_avg = sum(top1_recent) / float(len(top1_recent))
                ema_s = float(top1_ema) if top1_ema is not None else float("nan")
                print(
                    f"[rfp train] epoch={epoch} batch={bi + 1}/{n_micro} "
                    f"loss_micro={float(loss.detach()):.4f} "
                    f"top1_pct_batch={top1_batch:.2f}% "
                    f"top1_pct_roll_avg_last_{len(top1_recent)}={roll_avg:.2f}% "
                    f"top1_pct_ema_beta_{top1_ema_beta:g}={ema_s:.2f}% "
                    f"top1_pct_epoch_avg={top1_epoch_mean:.2f}% "
                    f"run={meta}",
                    flush=True,
                )

        scheduler.step()
        epochs_ran = epoch
        last_train_loss = train_loss / max(n_batches, 1)
        last_avg_train = {"from_sq_top1": sum_top1 / max(n_batches, 1)}

        model.eval()
        val_loss = 0.0
        val_top1_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for delta_z, z_curr, hist_mask, from_m, fs, elo_bucket in val_loader:
                loss, m = _forward_rfp_batch(
                    model,
                    delta_z,
                    z_curr,
                    hist_mask,
                    elo_bucket,
                    from_m,
                    fs,
                    label_smoothing=label_smoothing,
                    elo_null_prob=0.0,
                    train=False,
                    device=device,
                    use_amp=use_amp,
                )
                val_loss += float(loss.detach())
                val_top1_sum += float(m["top1"])
                val_batches += 1

        va_loss = val_loss / max(val_batches, 1)
        va_top1 = val_top1_sum / max(val_batches, 1)
        last_val = {"val_loss": va_loss, "val_from_sq_top1": va_top1}
        print(
            f"[rfp epoch {epoch}/{epochs}] train_loss={last_train_loss:.4f} "
            f"train_top1_pct={last_avg_train['from_sq_top1']:.2f}% "
            f"val_loss={va_loss:.4f} val_top1_pct={va_top1:.2f}%",
            flush=True,
        )

        if va_loss < best_val:
            best_val = va_loss
            best_ep = epoch

        if early_thr is not None:
            thr = float(early_thr)
            if math.isfinite(thr) and last_avg_train["from_sq_top1"] / 100.0 >= thr:
                print(
                    f"[early_stop] epoch={epoch} train_top1_pct={last_avg_train['from_sq_top1']:.2f}% "
                    f">= threshold_pct={thr * 100.0:.2f}%",
                    flush=True,
                )
                early_stopped = True
                break

    return best_val, best_ep, last_val, last_train_loss, last_avg_train, epochs_ran, early_stopped
