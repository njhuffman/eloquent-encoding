"""Epoch loop for Chess-JEPA (triplet + VICReg + EMA target)."""

from __future__ import annotations

import sys

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from jepa.architectures import jepa_triplet_vicreg_loss


def run_training_epochs(
    model: torch.nn.Module,
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    use_amp: bool,
    ema_momentum: float,
    margin_alpha: float,
    vicreg_var_coef: float,
    vicreg_std_target: float,
    log_interval: int,
) -> tuple[float, int]:
    """
    Train for ``epochs``. Returns (best_val_loss, best_epoch).
    Updates ``model`` in place; EMA target updated each step.
    """
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    best_val = float("inf")
    best_ep = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        sum_pct_pos = 0.0
        sum_pct_triplet_inactive = 0.0
        sum_vicreg_std_mean = 0.0
        for bi, batch in enumerate(train_loader):
            board_t, pos, negs, elo = batch
            board_t = board_t.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            negs = negs.to(device, non_blocking=True)
            elo = elo.to(device, non_blocking=True)
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    z_online, z_hat = model.forward_online(board_t, elo)
                    with torch.no_grad():
                        z_pos = model.forward_target(pos)
                        z_negs = model.forward_target_stack(negs)
                    loss, m = jepa_triplet_vicreg_loss(
                        z_online,
                        z_hat,
                        z_pos,
                        z_negs,
                        margin_alpha=margin_alpha,
                        vicreg_var_coef=vicreg_var_coef,
                        vicreg_std_target=vicreg_std_target,
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                z_online, z_hat = model.forward_online(board_t, elo)
                with torch.no_grad():
                    z_pos = model.forward_target(pos)
                    z_negs = model.forward_target_stack(negs)
                loss, m = jepa_triplet_vicreg_loss(
                    z_online,
                    z_hat,
                    z_pos,
                    z_negs,
                    margin_alpha=margin_alpha,
                    vicreg_var_coef=vicreg_var_coef,
                    vicreg_std_target=vicreg_std_target,
                )
                loss.backward()
                optimizer.step()
            model.ema_update_target(ema_momentum)
            train_loss += float(loss.detach())
            n_batches += 1
            sum_pct_pos += m["pct_pos_beats_all_negs"]
            sum_pct_triplet_inactive += m["pct_triplet_inactive"]
            sum_vicreg_std_mean += m["vicreg_std_mean"]
            if log_interval > 0 and (bi + 1) % log_interval == 0:
                print(
                    f"Epoch {epoch} [{bi + 1}/{len(train_loader)}] loss={loss.item():.4f} "
                    f"pos_beats_all_negs={m['pct_pos_beats_all_negs']:.1f}% "
                    f"triplet_inactive={m['pct_triplet_inactive']:.1f}% "
                    f"std_mean={m['vicreg_std_mean']:.4f} (vicreg_std_target={vicreg_std_target:.4f})",
                    file=sys.stderr,
                )
        train_loss /= max(n_batches, 1)
        avg_pct_pos = sum_pct_pos / max(n_batches, 1)
        avg_pct_trip = sum_pct_triplet_inactive / max(n_batches, 1)
        avg_vicreg_std_mean = sum_vicreg_std_mean / max(n_batches, 1)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        v_batches = 0
        val_sum_pct_pos = 0.0
        val_sum_pct_triplet_inactive = 0.0
        val_sum_vicreg_std_mean = 0.0
        with torch.no_grad():
            for batch in val_loader:
                board_t, pos, negs, elo = batch
                board_t = board_t.to(device, non_blocking=True)
                pos = pos.to(device, non_blocking=True)
                negs = negs.to(device, non_blocking=True)
                elo = elo.to(device, non_blocking=True)
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        z_online, z_hat = model.forward_online(board_t, elo)
                        z_pos = model.forward_target(pos)
                        z_negs = model.forward_target_stack(negs)
                        loss, vm = jepa_triplet_vicreg_loss(
                            z_online,
                            z_hat,
                            z_pos,
                            z_negs,
                            margin_alpha=margin_alpha,
                            vicreg_var_coef=vicreg_var_coef,
                            vicreg_std_target=vicreg_std_target,
                        )
                else:
                    z_online, z_hat = model.forward_online(board_t, elo)
                    z_pos = model.forward_target(pos)
                    z_negs = model.forward_target_stack(negs)
                    loss, vm = jepa_triplet_vicreg_loss(
                        z_online,
                        z_hat,
                        z_pos,
                        z_negs,
                        margin_alpha=margin_alpha,
                        vicreg_var_coef=vicreg_var_coef,
                        vicreg_std_target=vicreg_std_target,
                    )
                val_loss += float(loss.detach())
                v_batches += 1
                val_sum_pct_pos += vm["pct_pos_beats_all_negs"]
                val_sum_pct_triplet_inactive += vm["pct_triplet_inactive"]
                val_sum_vicreg_std_mean += vm["vicreg_std_mean"]
        val_loss /= max(v_batches, 1)
        val_avg_pct_pos = val_sum_pct_pos / max(v_batches, 1)
        val_avg_pct_trip = val_sum_pct_triplet_inactive / max(v_batches, 1)
        val_avg_vicreg_std_mean = val_sum_vicreg_std_mean / max(v_batches, 1)
        print(
            f"Epoch {epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train pos_beats_all_negs={avg_pct_pos:.1f}% triplet_inactive={avg_pct_trip:.1f}% "
            f"std_mean={avg_vicreg_std_mean:.4f} "
            f"val pos_beats_all_negs={val_avg_pct_pos:.1f}% triplet_inactive={val_avg_pct_trip:.1f}% "
            f"std_mean={val_avg_vicreg_std_mean:.4f} "
            f"(vicreg_std_target={vicreg_std_target:.4f})",
            file=sys.stderr,
        )
        if val_loss < best_val:
            best_val = val_loss
            best_ep = epoch

    return best_val, best_ep


def save_submodule_sidecars(checkpoint_path: Path, model: torch.nn.Module) -> None:
    stem = checkpoint_path.stem
    parent = checkpoint_path.parent
    torch.save(model.encoder_online.state_dict(), parent / f"{stem}_encoder_online.pt")
    torch.save(model.encoder_target.state_dict(), parent / f"{stem}_encoder_target.pt")
    torch.save(model.predictor.state_dict(), parent / f"{stem}_predictor.pt")
