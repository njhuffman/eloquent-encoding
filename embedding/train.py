#!/usr/bin/env python3
"""
Train the chess MAE: load train/val HDF5, run training loop, checkpoint best model.
"""

import argparse
import sys
from pathlib import Path

import torch

# Add repo root for imports when run as script
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from embedding.config import (
    BATCH_SIZE,
    BEST_CHECKPOINT_NAME,
    CHECKPOINT_DIR,
    LEARNING_RATE,
    LOG_INTERVAL,
    NUM_EPOCHS,
    DATALOADER_NUM_WORKERS,
)
from embedding.dataset import get_dataloaders
from embedding.model import ChessMAE, masked_mse_loss


def main() -> int:
    parser = argparse.ArgumentParser(description="Train chess board MAE")
    parser.add_argument("--train-h5", type=Path, required=True, help="Train split HDF5 (e.g. train.h5)")
    parser.add_argument("--val-h5", type=Path, required=True, help="Val split HDF5")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size (default {BATCH_SIZE})")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help=f"Epochs (default {NUM_EPOCHS})")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help=f"Learning rate (default {LEARNING_RATE})")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path(CHECKPOINT_DIR), help="Where to save checkpoints")
    parser.add_argument("--workers", type=int, default=DATALOADER_NUM_WORKERS, help="DataLoader workers")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda, cuda:0, cpu, or leave unset to auto-select (prefer GPU)",
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed-precision training (slower on GPU)")
    args = parser.parse_args()

    if not args.train_h5.exists():
        print(f"Error: train HDF5 not found: {args.train_h5}", file=sys.stderr)
        return 1
    if not args.val_h5.exists():
        print(f"Error: val HDF5 not found: {args.val_h5}", file=sys.stderr)
        return 1

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Error: --device cuda requested but CUDA is not available (no NVIDIA GPU/driver?).", file=sys.stderr)
        return 1
    use_amp = not args.no_amp and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device} (mixed precision: {use_amp})", file=sys.stderr)

    train_loader, val_loader = get_dataloaders(
        args.train_h5,
        args.val_h5,
        batch_size=args.batch_size,
        num_workers=args.workers,
        val_seed=0,
    )

    model = ChessMAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
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
            if (bi + 1) % LOG_INTERVAL == 0:
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
            ckpt_path = args.checkpoint_dir / BEST_CHECKPOINT_NAME
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                ckpt_path,
            )
            print(f"  -> saved best checkpoint to {ckpt_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
