"""Epoch training for jepa2 (streaming legals + CE + MSE + VICReg + EMA)."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from jepa.checkpoint_utils import build_model_checkpoint
from jepa2.architectures import resolve_config_for_id
from jepa2.checkpoint_paths import stage_checkpoint_path
from jepa2.chess_io import prepare_batch_tensors
from jepa2.loss import jepa2_loss_forward
from jepa2.load import load_jepa2_from_checkpoint


def _format_train_batch_log(epoch: int, bi: int, loss_f: float, m: dict[str, float]) -> str:
    """One-line stderr log: total loss plus CE / MSE / VICReg breakdown for this batch."""
    return (
        f"epoch {epoch} batch {bi} "
        f"loss={loss_f:.4f} "
        f"ce={m.get('ce', float('nan')):.4f} ce_w={m.get('ce_weighted', float('nan')):.4f} "
        f"mse={m.get('mse_played', float('nan')):.6f} mse_rms={m.get('mse_hat_pos_rms', float('nan')):.5f} "
        f"mse_w={m.get('mse_played_weighted', float('nan')):.4f} "
        f"vic_w={m.get('vicreg_weighted', float('nan')):.4f} "
        f"vic_std_mean={m.get('vicreg_std_mean', float('nan')):.4f} "
        f"vic_std_min={m.get('vicreg_std_min', float('nan')):.4f} vic_std_max={m.get('vicreg_std_max', float('nan')):.4f} "
        f"vic_std_tgt={m.get('vicreg_std_target', float('nan')):.3f} vic_var_pen={m.get('vicreg_var', float('nan')):.5f} "
        f"vic_inv={m.get('vicreg_inv', float('nan')):.5f} vic_cov={m.get('vicreg_cov', float('nan')):.5f} "
        f"top1={m.get('top1_acc', float('nan')):.1f}%"
    )


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

    if use_amp:
        with torch.amp.autocast("cuda"):
            z_online, z_hat = model.forward_online(board_t, elo)
            z_all = model.forward_target_stack(succ)
    else:
        z_online, z_hat = model.forward_online(board_t, elo)
        z_all = model.forward_target_stack(succ)

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
        mse_played_weight=float(resolved["mse_played_weight"]),
        ce_label_smoothing=float(resolved["ce_label_smoothing"]),
        vicreg=dict(resolved["vicreg"]),
        use_amp_cuda=bool(use_amp and device.type == "cuda"),
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
) -> dict[str, float]:
    """One forward-only pass over train (train mode) and val (eval)."""
    sums: dict[str, float] = {}
    counts = {"train": 0, "val": 0}

    def _accum(split: str, m: dict[str, float]) -> None:
        counts[split] += 1
        for k, v in m.items():
            sums[f"{split}_{k}"] = sums.get(f"{split}_{k}", 0.0) + v

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

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    best_val = float("inf")
    best_ep = 0
    last_inf: dict[str, float] = {}
    last_train_epoch_loss = 0.0
    last_avg_train: dict[str, float] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        sum_metrics: dict[str, float] = {}

        for bi, (board_t, elo, fs, ts, pr, fens) in enumerate(train_loader):
            rng = random.Random(global_step_seed + epoch * 1_000_000 + bi)
            optimizer.zero_grad()
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
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
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
                loss.backward()
                optimizer.step()

            model.ema_update_target(ema_momentum)
            train_loss += float(loss.detach())
            n_batches += 1
            for k, v in m.items():
                sum_metrics[k] = sum_metrics.get(k, 0.0) + v

            if log_interval and bi % log_interval == 0:
                print(_format_train_batch_log(epoch, bi, float(loss), m), file=sys.stderr)

        train_loss /= max(n_batches, 1)
        avg_train = {k: sum_metrics[k] / max(n_batches, 1) for k in sum_metrics}

        inf = compute_epoch_metrics_inference(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            resolved=resolved,
            use_amp=use_amp,
            val_seed=int(resolved.get("val_legal_seed", 42)),
            epoch=epoch,
        )
        val_loss = inf.get("val_loss", train_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_ep = epoch

        scheduler.step()
        last_inf = inf
        last_train_epoch_loss = train_loss
        last_avg_train = avg_train

    return best_val, best_ep, last_inf, last_train_epoch_loss, last_avg_train


def write_stage_metrics_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")


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
