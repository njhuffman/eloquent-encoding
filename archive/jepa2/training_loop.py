"""Epoch training for jepa2 (streaming legals + CE + VICReg + EMA)."""

from __future__ import annotations

import json
import math
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from jepa.checkpoint_utils import build_model_checkpoint
from jepa2.architectures import resolve_config_for_id
from jepa2.checkpoint_paths import stage_checkpoint_path
from jepa2.chess_io import prepare_batch_tensors
from jepa2.gsnr_probe import flatten_encoder_online_grads, gsnr_metrics_from_grad_vectors
from jepa2.loss import jepa2_loss_forward
from jepa2.sam import sam_apply_perturbation, sam_build_perturbations, sam_revert_perturbation
from jepa2.load import load_jepa2_from_checkpoint


_LOG_METRIC_ORDER = (
    "loss",
    "ce",
    "ce_weighted",
    "ce_temperature",
    "ce_logit_min",
    "ce_logit_max",
    "ce_logit_true_mean",
    "vicreg_weighted",
    "vicreg_inv",
    "vicreg_inv_rms",
    "vicreg_std_mean",
    "vicreg_std_min",
    "vicreg_std_max",
    "vicreg_cov",
    "succ_vicreg_weighted",
    "succ_vicreg_var",
    "succ_vicreg_cov",
    "succ_vicreg_std_mean",
    "top1_acc",
    "softmax_entropy",
    "mean_n_legals_scored",
)


def _metric_triples_from_window(ms: list[dict[str, float]]) -> dict[str, tuple[float, float, float]]:
    """Per key: (mean, min, max) over micro-batches in one accumulation window."""
    if not ms:
        return {}
    keys: set[str] = set()
    for m in ms:
        keys.update(m.keys())
    out: dict[str, tuple[float, float, float]] = {}
    for k in keys:
        vals = [float(m[k]) for m in ms if k in m]
        vals = [v for v in vals if v == v]  # finite only
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
    last_gsnr_within: float | None = None,
    last_gsnr_between: float | None = None,
) -> str:
    """One-line stderr log after an optimizer step: mean/min/max per metric over the window."""
    lm, ll, lh = triples.get("loss", (float("nan"), float("nan"), float("nan")))
    parts = [
        f"epoch {epoch} opt_step {completed_opt_steps} micro_batches {micro_lo}-{micro_hi} accum={accum_steps}",
        f"loss_avg={lm:.4f} loss_min={ll:.4f} loss_max={lh:.4f}",
    ]
    seen: set[str] = {"loss"}
    for k in _LOG_METRIC_ORDER:
        if k == "loss" or k not in triples:
            continue
        seen.add(k)
        mn, lo, hi = triples[k]
        if k == "top1_acc":
            parts.append(f"{k}_avg={mn:.2f}% {k}_min={lo:.2f}% {k}_max={hi:.2f}%")
        else:
            parts.append(f"{k}_avg={mn:.4f} {k}_min={lo:.4f} {k}_max={hi:.4f}")
    for k in sorted(triples.keys()):
        if k in seen:
            continue
        mn, lo, hi = triples[k]
        parts.append(f"{k}_avg={mn:.4f} {k}_min={lo:.4f} {k}_max={hi:.4f}")
    if last_gsnr_within is not None and math.isfinite(last_gsnr_within):
        parts.append(f"gsnr_within_last={last_gsnr_within:.4e}")
    if last_gsnr_between is not None and math.isfinite(last_gsnr_between):
        parts.append(f"gsnr_between_last={last_gsnr_between:.4e}")
    return " ".join(parts)


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
    """K extra forward-backward passes; no optimizer step. Encoder grads only for GSNR."""
    if not hasattr(model, "encoder_online"):
        return {
            "gsnr_encoder": float("nan"),
            "gsnr_signal": float("nan"),
            "gsnr_noise": float("nan"),
            "gsnr_grad_norm_mean": float("nan"),
        }
    grads_cpu: list[torch.Tensor] = []
    for tup in buffer:
        board_t, elo, fs, ts, pr, fens, ep, micro_bi = tup
        board_t = board_t.to(device, non_blocking=True)
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
                    elo,
                    fs,
                    ts,
                    pr,
                    fens,
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
                elo,
                fs,
                ts,
                pr,
                fens,
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
    """Flattened encoder grad from accumulated ``.grad``; divide by AMP scale when active."""
    g = flatten_encoder_online_grads(model)
    if use_amp and scaler is not None:
        scale = float(scaler.get_scale())
        g = g / max(scale, 1e-8)
    return g.detach().cpu().clone()


def _between_step_grad_vectors_for_gsnr(
    step_grad_buf: deque,
    k: int,
) -> list[torch.Tensor] | None:
    """Last up to ``k`` consecutive aggregated step grads; need at least 2."""
    if len(step_grad_buf) < 2:
        return None
    lst = list(step_grad_buf)
    if len(lst) >= k:
        return lst[-k:]
    return lst


def _batch_to_rows(
    board_t: torch.Tensor,
    elo: torch.Tensor,
    fs: torch.Tensor,
    ts: torch.Tensor,
    pr: torch.Tensor,
    fens: list[str],
) -> list[dict[str, Any]]:
    B = int(board_t.shape[0])
    rows: list[dict[str, Any]] = []
    for i in range(B):
        rows.append(
            {
                "board_t": board_t[i].detach().cpu().numpy(),
                "elo_to_move": float(elo[i].item()),
                "from_sq": int(fs[i].item()),
                "to_sq": int(ts[i].item()),
                "promotion": int(pr[i].item()),
                "fen": fens[i],
            }
        )
    return rows


def _forward_batch(
    model: torch.nn.Module,
    board_t: torch.Tensor,
    elo: torch.Tensor,
    fs: torch.Tensor,
    ts: torch.Tensor,
    pr: torch.Tensor,
    fens: list[str],
    resolved: dict[str, Any],
    *,
    train: bool,
    rng: random.Random,
    device: torch.device,
    use_amp: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    M_cap = int(resolved["M_train"]) if train else int(resolved["M_eval"])
    rows = _batch_to_rows(board_t, elo, fs, ts, pr, fens)
    _, _, succ, mask, labels, _ = prepare_batch_tensors(rows, M_cap, rng)
    board_t = board_t.to(device, non_blocking=True)
    elo = elo.to(device, non_blocking=True)
    succ = succ.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    fs_dev = fs.to(device, non_blocking=True, dtype=torch.long)
    p_unk = float(resolved.get("from_sq_unknown_probability", 0.0))
    if train and p_unk > 0.0:
        from_sq_idx = fs_dev.clone()
        if p_unk >= 1.0:
            from_sq_idx.fill_(64)
        else:
            u = torch.rand(from_sq_idx.shape[0], device=device, dtype=torch.float32)
            from_sq_idx[u < p_unk] = 64
    else:
        from_sq_idx = fs_dev

    vr = dict(resolved["vicreg"])
    succ_vc = float(vr.get("succ_var_coef", 0.0))
    succ_cc = float(vr.get("succ_cov_coef", 0.0))
    need_succ_vic = succ_vc > 0.0 or succ_cc > 0.0

    if use_amp:
        with torch.amp.autocast("cuda"):
            z_online, z_hat = model.forward_online(board_t, elo, from_sq_idx)
            z_all = model.forward_target_stack(succ)
            z_online_legals = (
                model.forward_online_stack(succ) if need_succ_vic else None
            )
    else:
        z_online, z_hat = model.forward_online(board_t, elo, from_sq_idx)
        z_all = model.forward_target_stack(succ)
        z_online_legals = model.forward_online_stack(succ) if need_succ_vic else None

    b_idx = torch.arange(labels.shape[0], device=device, dtype=torch.long)
    z_pos = z_all[b_idx, labels]
    z_legals = z_all

    loss, metrics = jepa2_loss_forward(
        z_online,
        z_hat,
        z_pos,
        z_legals,
        mask,
        labels,
        ce_weight=float(resolved["ce_weight"]),
        ce_label_smoothing=float(resolved["ce_label_smoothing"]),
        ce_temperature=float(resolved["ce_temperature"]),
        vicreg=vr,
        use_amp_cuda=bool(use_amp and device.type == "cuda"),
        z_online_legals=z_online_legals,
    )
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
    """
    Forward-only metrics: optionally train (train mode), then val (eval).

    When ``include_train`` is False, skips the train pass (e.g. training loop
    already averaged per-batch train metrics).
    """
    sums: dict[str, float] = {}
    counts = {"train": 0, "val": 0}

    def _accum(split: str, m: dict[str, float]) -> None:
        counts[split] += 1
        for k, v in m.items():
            sums[f"{split}_{k}"] = sums.get(f"{split}_{k}", 0.0) + v

    if include_train:
        model.train()
        with torch.no_grad():
            for bi, (board_t, elo, fs, ts, pr, fens) in enumerate(train_loader):
                rng = random.Random(val_seed + epoch * 1_000_003 + bi)
                _, m = _forward_batch(
                    model,
                    board_t,
                    elo,
                    fs,
                    ts,
                    pr,
                    fens,
                    resolved,
                    train=True,
                    rng=rng,
                    device=device,
                    use_amp=use_amp,
                )
                _accum("train", m)

    model.eval()
    with torch.no_grad():
        for bi, (board_t, elo, fs, ts, pr, fens) in enumerate(val_loader):
            rng = random.Random(val_seed + epoch * 1_000_003 + 50_000 + bi)
            _, m = _forward_batch(
                model,
                board_t,
                elo,
                fs,
                ts,
                pr,
                fens,
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
) -> tuple[float, int, dict[str, float], float, dict[str, float]]:
    tr = resolved["train"]
    epochs = int(tr["epochs"])
    lr = float(tr["learning_rate"])
    wd = float(tr["weight_decay"])
    use_amp = bool(resolved.get("use_amp", True)) and device.type == "cuda"
    ema_momentum = float(resolved["ema_momentum"])
    log_interval = int(resolved.get("log_interval", 100))
    accum_steps = max(int(tr.get("gradient_accumulation_steps", 1)), 1)

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

        for bi, (board_t, elo, fs, ts, pr, fens) in enumerate(train_loader):
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
                        elo,
                        fs,
                        ts,
                        pr,
                        fens,
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
                    elo,
                    fs,
                    ts,
                    pr,
                    fens,
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
                        elo.detach().cpu(),
                        fs.detach().cpu(),
                        ts.detach().cpu(),
                        pr.detach().cpu(),
                        list(fens),
                        epoch,
                        bi,
                    )
                )

            micro_in_accum += 1
            is_last_micro = bi == n_micro_epoch - 1
            should_step = micro_in_accum == accum_steps or (is_last_micro and 0 < micro_in_accum < accum_steps)

            if should_step:
                replay_batches = list(replay_batch_window) if need_replay_buffer else []

                # Between-step GSNR uses first-pass encoder grad (before SAM unscale/perturb/replay).
                if gsnr_every > 0 and step_grad_buf is not None and hasattr(model, "encoder_online"):
                    step_grad_buf.append(
                        _aggregated_encoder_grad_vector(model, use_amp=use_amp, scaler=scaler)
                    )

                if sam_rho > 0.0:
                    trainables = list(model.trainable_parameters())
                    # Build epsilon from current .grad without scaler.unscale_: under AMP, grads are
                    # scaled by S but eps = rho * g / ||g|| equals rho * (g/S) / ||g/S||, so one
                    # scaler.step() can unscale the second-pass accumulation (GradScaler allows
                    # only one unscale_ per step).
                    perturbations = sam_build_perturbations(trainables, sam_rho)
                    if perturbations:
                        sam_apply_perturbation(perturbations)
                        optimizer.zero_grad(set_to_none=True)
                        for tup in replay_batches:
                            board_t, elo, fs, ts, pr, fens, ep, micro_bi = tup
                            board_t = board_t.to(device, non_blocking=True)
                            elo = elo.to(device, non_blocking=True)
                            fs = fs.to(device, non_blocking=True)
                            ts = ts.to(device, non_blocking=True)
                            pr = pr.to(device, non_blocking=True)
                            rng = random.Random(global_step_seed + int(ep) * 1_000_000 + int(micro_bi))
                            if use_amp:
                                assert scaler is not None
                                with torch.amp.autocast("cuda"):
                                    loss2, _ = _forward_batch(
                                        model,
                                        board_t,
                                        elo,
                                        fs,
                                        ts,
                                        pr,
                                        fens,
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
                                    board_t,
                                    elo,
                                    fs,
                                    ts,
                                    pr,
                                    fens,
                                    resolved,
                                    train=True,
                                    rng=rng,
                                    device=device,
                                    use_amp=False,
                                )
                                (loss2 / float(accum_steps)).backward()
                        sam_revert_perturbation(perturbations)
                    if use_amp:
                        assert scaler is not None
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                else:
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

                    print(
                        f"[gsnr] epoch={epoch} opt_step={completed_opt_steps} "
                        f"within_update={_fmt_gsnr(within_g)} signal_w={_fmt_gsnr(ws)} noise_w={_fmt_gsnr(wn)} "
                        f"grad_norm_mean_w={_fmt_gsnr(wgmn)} "
                        f"between_updates={_fmt_gsnr(between_g)} signal_b={_fmt_gsnr(bs)} noise_b={_fmt_gsnr(bn)} "
                        f"grad_norm_mean_b={_fmt_gsnr(bgmn)}",
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
                            last_gsnr_within=last_gsnr_within,
                            last_gsnr_between=last_gsnr_between,
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

    return best_val, best_ep, last_inf, last_train_epoch_loss, last_avg_train


def write_stage_metrics_json(path: Path, record: dict[str, Any]) -> Path:
    """
    Write per-stage metrics JSON. Returns the path actually written.

    If ``path`` is not writable (e.g. shared bulk volume owned by root), falls back to
    ``{repo}/jepa2_checkpoints/{model}/metrics/{same filename}`` so training can finish.
    """
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
        alt = repo_root / "jepa2_checkpoints" / name.strip() / "metrics" / path.name
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
        extra={"resolved_training": resolved, "jepa2": True},
    )
    torch.save(payload, out_path)
    return out_path


def init_stage_zero(spec: dict[str, Any], device: torch.device) -> Path:
    from jepa2.architectures import build_model

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
