#!/usr/bin/env python3
"""
Train Chess-JEPA from a JSON training spec (see jepa/model_configs/).

CLI: --model <name> or --config path.json; overrides for device, workers, dirs, AMP, register.
"""

from __future__ import annotations

import argparse
import copy
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from jepa.h5_bootstrap import apply_hdf5_read_safety_env

apply_hdf5_read_safety_env()

import h5py
import numpy as np
import torch

from jepa.architectures import build_model, jepa_triplet_vicreg_loss, resolve_config_for_id
from jepa.checkpoint_utils import build_model_checkpoint
from jepa.config import (
    BEST_CHECKPOINT_NAME,
    BOARD_CHANNELS,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    MODEL_CONFIGS_DIR,
)
from jepa.dataset import assert_h5_k_matches, get_dataloaders, h5_transition_counts
from jepa.registry import register_model
from jepa.training_spec import apply_runtime_overrides, load_training_spec


def _write_synthetic_jepa_h5(path: Path, n_rows: int, k_neg: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["num_negatives_k"] = k_neg
        bt = rng.random((n_rows, BOARD_HEIGHT, BOARD_WIDTH, BOARD_CHANNELS), dtype=np.float32)
        pos = rng.random((n_rows, BOARD_HEIGHT, BOARD_WIDTH, BOARD_CHANNELS), dtype=np.float32)
        negs = rng.random((n_rows, k_neg, BOARD_HEIGHT, BOARD_WIDTH, BOARD_CHANNELS), dtype=np.float32)
        elo = rng.uniform(800.0, 2800.0, size=(n_rows,)).astype(np.float32)
        f.create_dataset("board_t", data=bt)
        f.create_dataset("board_t_plus_1_pos", data=pos)
        f.create_dataset("board_t_plus_1_negs", data=negs)
        f.create_dataset("elo", data=elo)


def _smoke_run(device: torch.device) -> int:
    k_neg = 4
    d_model = 64
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        train_h5 = td_path / "train.h5"
        val_h5 = td_path / "val.h5"
        _write_synthetic_jepa_h5(train_h5, n_rows=32, k_neg=k_neg)
        _write_synthetic_jepa_h5(val_h5, n_rows=16, k_neg=k_neg)

        arch_cfg = {
            "d_model": d_model,
            "encoder_layers": 2,
            "predictor_layers": 1,
            "nhead": 4,
            "dim_feedforward": 128,
            "dropout": 0.0,
            "use_cls": True,
            "elo_scale": 3000.0,
            "num_negatives_k": k_neg,
        }
        model = build_model("chess_jepa_v1", arch_cfg).to(device)
        opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4, weight_decay=0.05)
        train_loader, val_loader = get_dataloaders(
            train_h5,
            val_h5,
            batch_size=8,
            num_workers=0,
            in_memory=True,
        )
        margin = 0.2
        vic_c = 0.1
        vic_t = 1.0
        ema_m = 0.99

        model.train()
        batch = next(iter(train_loader))
        board_t, pos, negs, elo = [x.to(device) for x in batch]
        z_o, z_h = model.forward_online(board_t, elo)
        with torch.no_grad():
            z_pos = model.forward_target(pos)
            z_negs = model.forward_target_stack(negs)
        loss, _ = jepa_triplet_vicreg_loss(
            z_o, z_h, z_pos, z_negs, margin_alpha=margin, vicreg_var_coef=vic_c, vicreg_std_target=vic_t
        )
        loss.backward()
        opt.step()
        opt.zero_grad()
        model.ema_update_target(ema_m)

        model.eval()
        with torch.no_grad():
            vb = next(iter(val_loader))
            board_t, pos, negs, elo = [x.to(device) for x in vb]
            z_o, z_h = model.forward_online(board_t, elo)
            z_pos = model.forward_target(pos)
            z_negs = model.forward_target_stack(negs)
            vloss, _ = jepa_triplet_vicreg_loss(
                z_o, z_h, z_pos, z_negs, margin_alpha=margin, vicreg_var_coef=vic_c, vicreg_std_target=vic_t
            )
        ckpt_path = td_path / "smoke_best.pt"
        payload = build_model_checkpoint(
            model,
            architecture_id="chess_jepa_v1",
            architecture_config=arch_cfg,
            train_meta={"n_train_boards": 32, "n_val_boards": 16},
            train_hparams={"best_val_loss": float(vloss.detach()), "epoch": 1},
            training_spec={"name": "smoke"},
        )
        torch.save(payload, ckpt_path)
        assert ckpt_path.is_file()
    print("jepa smoke: ok", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train Chess-JEPA from a JSON training spec.",
        epilog=f"Specs live under {_REPO_ROOT / MODEL_CONFIGS_DIR} as <name>.json",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny synthetic train/val step and checkpoint write; ignore other args.",
    )
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument(
        "--model",
        type=str,
        metavar="NAME",
        help=f"Load {MODEL_CONFIGS_DIR}/<NAME>.json",
    )
    src.add_argument("--config", type=Path, help="Path to training spec JSON")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override training.dataloader_num_workers from the spec",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="torch device (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Override outputs.checkpoint_dir from the spec",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=None,
        help="Override outputs.artifacts_dir",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Override training.use_amp to false",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register best checkpoint after training",
    )
    args = parser.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Error: --device cuda requested but CUDA is not available.", file=sys.stderr)
        return 1

    if args.smoke:
        return _smoke_run(device)

    if not args.model and not args.config:
        print("Error: provide --model NAME or --config path.json (or use --smoke).", file=sys.stderr)
        return 1

    try:
        if args.config is not None:
            base_spec, _src_path = load_training_spec(repo_root=_REPO_ROOT, config_path=args.config)
        else:
            base_spec, _src_path = load_training_spec(repo_root=_REPO_ROOT, model_name=args.model)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    use_amp_override = False if args.no_amp else None
    effective = apply_runtime_overrides(
        base_spec,
        workers=args.workers,
        checkpoint_dir=args.checkpoint_dir,
        artifacts_dir=args.artifacts_dir,
        use_amp=use_amp_override,
        register=None,
        repo_root=_REPO_ROOT,
    )

    train_p = Path(effective["data"]["train_h5"])
    val_p = Path(effective["data"]["val_h5"])
    if not train_p.is_file():
        print(f"Error: train HDF5 not found: {train_p}", file=sys.stderr)
        return 1
    if not val_p.is_file():
        print(f"Error: val HDF5 not found: {val_p}", file=sys.stderr)
        return 1

    arch_id = effective["architecture"]["id"]
    arch_cfg_user = effective["architecture"]["config"]
    resolved_architecture_config = resolve_config_for_id(arch_id, arch_cfg_user)
    k_expected = int(resolved_architecture_config["num_negatives_k"])
    try:
        assert_h5_k_matches(train_p, k_expected)
        assert_h5_k_matches(val_p, k_expected)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    ckpt_dir = Path(effective["outputs"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tr = effective["training"]
    use_amp = bool(tr["use_amp"]) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Model {effective['name']!r} | device: {device} | AMP: {use_amp}", file=sys.stderr)

    n_train, n_val = h5_transition_counts(train_p, val_p)
    train_meta = {
        "n_train_boards": int(n_train),
        "n_val_boards": int(n_val),
        "train_h5": str(train_p),
        "val_h5": str(val_p),
        "train_h5_basename": train_p.name,
        "val_h5_basename": val_p.name,
    }

    train_loader, val_loader = get_dataloaders(
        train_p,
        val_p,
        batch_size=tr["batch_size"],
        num_workers=tr["dataloader_num_workers"],
        in_memory=tr["in_memory"],
    )

    model = build_model(arch_id, arch_cfg_user).to(device)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=tr["learning_rate"],
        weight_decay=tr["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tr["epochs"])
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    best_val_loss = float("inf")
    log_interval = tr["log_interval"]
    ema_m = float(tr["ema_momentum"])
    margin = float(tr["triplet_margin_alpha"])
    vic_c = float(tr["vicreg_var_coef"])
    vic_t = float(tr["vicreg_std_target"])

    spec_for_ckpt = copy.deepcopy(effective)
    spec_for_ckpt["runtime"] = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }

    for epoch in range(1, tr["epochs"] + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
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
                    loss, _ = jepa_triplet_vicreg_loss(
                        z_online,
                        z_hat,
                        z_pos,
                        z_negs,
                        margin_alpha=margin,
                        vicreg_var_coef=vic_c,
                        vicreg_std_target=vic_t,
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                z_online, z_hat = model.forward_online(board_t, elo)
                with torch.no_grad():
                    z_pos = model.forward_target(pos)
                    z_negs = model.forward_target_stack(negs)
                loss, _ = jepa_triplet_vicreg_loss(
                    z_online,
                    z_hat,
                    z_pos,
                    z_negs,
                    margin_alpha=margin,
                    vicreg_var_coef=vic_c,
                    vicreg_std_target=vic_t,
                )
                loss.backward()
                optimizer.step()
            model.ema_update_target(ema_m)
            train_loss += float(loss.detach())
            n_batches += 1
            if log_interval > 0 and (bi + 1) % log_interval == 0:
                print(
                    f"Epoch {epoch} [{bi + 1}/{len(train_loader)}] loss={loss.item():.4f}",
                    file=sys.stderr,
                )
        train_loss /= max(n_batches, 1)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        v_batches = 0
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
                        loss, _ = jepa_triplet_vicreg_loss(
                            z_online,
                            z_hat,
                            z_pos,
                            z_negs,
                            margin_alpha=margin,
                            vicreg_var_coef=vic_c,
                            vicreg_std_target=vic_t,
                        )
                else:
                    z_online, z_hat = model.forward_online(board_t, elo)
                    z_pos = model.forward_target(pos)
                    z_negs = model.forward_target_stack(negs)
                    loss, _ = jepa_triplet_vicreg_loss(
                        z_online,
                        z_hat,
                        z_pos,
                        z_negs,
                        margin_alpha=margin,
                        vicreg_var_coef=vic_c,
                        vicreg_std_target=vic_t,
                    )
                val_loss += float(loss.detach())
                v_batches += 1
        val_loss /= max(v_batches, 1)

        print(f"Epoch {epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}", file=sys.stderr)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = ckpt_dir / BEST_CHECKPOINT_NAME
            train_hparams = {
                "batch_size": tr["batch_size"],
                "epochs": tr["epochs"],
                "learning_rate": tr["learning_rate"],
                "weight_decay": tr["weight_decay"],
                "in_memory": tr["in_memory"],
                "log_interval": tr["log_interval"],
                "use_amp": tr["use_amp"],
                "dataloader_num_workers": tr["dataloader_num_workers"],
                "ema_momentum": ema_m,
                "triplet_margin_alpha": margin,
                "vicreg_var_coef": vic_c,
                "vicreg_std_target": vic_t,
                "epoch": epoch,
                "best_val_loss": val_loss,
            }
            payload = build_model_checkpoint(
                model,
                architecture_id=arch_id,
                architecture_config=resolved_architecture_config,
                train_meta=train_meta,
                train_hparams=train_hparams,
                optimizer_state_dict=optimizer.state_dict(),
                epoch=epoch,
                val_loss=val_loss,
                training_spec=spec_for_ckpt,
            )
            torch.save(payload, ckpt_path)
            print(f"  -> saved best checkpoint to {ckpt_path}", file=sys.stderr)

    do_register = bool(effective["outputs"]["register"]) or args.register
    if do_register:
        ckpt_path = ckpt_dir / BEST_CHECKPOINT_NAME
        if not ckpt_path.is_file():
            print(f"Error: register requested but no checkpoint at {ckpt_path}", file=sys.stderr)
            return 1
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        reg_name = effective["name"]
        art = effective["outputs"]["artifacts_dir"]
        register_model(
            name=reg_name,
            architecture_id=state["architecture_id"],
            architecture_config=state["architecture_config"],
            train_meta=state["train_meta"],
            train_hparams=state["train_hparams"],
            checkpoint_payload=state,
            repo_root=_REPO_ROOT,
            artifacts_dir=art,
            training_spec=state.get("training_spec"),
        )
        print(f"Registered model as {reg_name!r}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
