#!/usr/bin/env python3
"""
Train the chess MAE from a JSON training spec (see embedding/model_configs/).

CLI: choose spec via --model <name> or --config path.json; override only runtime
(device, workers, checkpoint/artifacts dirs, AMP, register).
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import h5py
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from embedding.architectures import build_model, resolve_config_for_id
from embedding.checkpoint_utils import build_model_checkpoint
from embedding.config import ARTIFACTS_DIR, BEST_CHECKPOINT_NAME, MODEL_CONFIGS_DIR
from embedding.dataset import get_dataloaders
from embedding.model import masked_mse_loss
from embedding.registry import register_model
from embedding.training_spec import apply_runtime_overrides, load_training_spec


def _h5_board_counts(train_h5: Path, val_h5: Path) -> tuple[int, int]:
    with h5py.File(train_h5, "r") as f:
        n_train = f["board"].shape[0]
    with h5py.File(val_h5, "r") as f:
        n_val = f["board"].shape[0]
    return n_train, n_val


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train chess MAE from a JSON training spec.",
        epilog=f"Specs live under {_REPO_ROOT / MODEL_CONFIGS_DIR} as <name>.json",
    )
    src = parser.add_mutually_exclusive_group(required=True)
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
        help=f"Override outputs.artifacts_dir (default from spec or {ARTIFACTS_DIR})",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Override training.use_amp to false",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register best checkpoint after training (in addition to spec outputs.register if set)",
    )
    args = parser.parse_args()

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

    ckpt_dir = Path(effective["outputs"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Error: --device cuda requested but CUDA is not available.", file=sys.stderr)
        return 1

    tr = effective["training"]
    use_amp = bool(tr["use_amp"]) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Model {effective['name']!r} | device: {device} | AMP: {use_amp}", file=sys.stderr)

    arch_id = effective["architecture"]["id"]
    arch_cfg_user = effective["architecture"]["config"]
    resolved_architecture_config = resolve_config_for_id(arch_id, arch_cfg_user)

    m = effective["masking"]
    n_train, n_val = _h5_board_counts(train_p, val_p)
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
        val_seed=tr["val_seed"],
        in_memory=tr["in_memory"],
        min_mask_ratio=m["min_mask_ratio"],
        max_mask_ratio=m["max_mask_ratio"],
    )

    model = build_model(arch_id, arch_cfg_user).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tr["learning_rate"])
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    best_val_loss = float("inf")
    log_interval = tr["log_interval"]

    spec_for_ckpt = copy.deepcopy(effective)
    spec_for_ckpt["runtime"] = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }

    for epoch in range(1, tr["epochs"] + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for bi, (enc, mask, target) in enumerate(train_loader):
            enc = enc.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    _, pred = model(enc, mask)
                    loss = masked_mse_loss(pred, target, mask)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                _, pred = model(enc, mask)
                loss = masked_mse_loss(pred, target, mask)
                loss.backward()
                optimizer.step()
            train_loss += loss.item()
            n_batches += 1
            if log_interval > 0 and (bi + 1) % log_interval == 0:
                print(
                    f"Epoch {epoch} [{bi + 1}/{len(train_loader)}] loss={loss.item():.4f}",
                    file=sys.stderr,
                )
        train_loss /= max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        v_batches = 0
        with torch.no_grad():
            for enc, mask, target in val_loader:
                enc = enc.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        _, pred = model(enc, mask)
                        loss = masked_mse_loss(pred, target, mask)
                else:
                    _, pred = model(enc, mask)
                    loss = masked_mse_loss(pred, target, mask)
                val_loss += loss.item()
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
                "val_seed": tr["val_seed"],
                "in_memory": tr["in_memory"],
                "log_interval": tr["log_interval"],
                "use_amp": tr["use_amp"],
                "dataloader_num_workers": tr["dataloader_num_workers"],
                "min_mask_ratio": m["min_mask_ratio"],
                "max_mask_ratio": m["max_mask_ratio"],
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
