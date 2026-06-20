#!/usr/bin/env python3
"""
Smoke test: run a few training/val steps on GPU if available, else CPU.
Use this to verify the training loop and AMP work with no errors.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from embedding.dataset import get_dataloaders
from embedding.model import ChessMAE, masked_mse_loss


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Use existing HDF5 if present, else skip with message
    train_h5 = _REPO_ROOT / "embedding" / "data" / "train.h5"
    val_h5 = _REPO_ROOT / "embedding" / "data" / "val.h5"
    if not train_h5.exists() or not val_h5.exists():
        print("Skip: embedding/data/train.h5 or val.h5 not found. Run pgn_to_hdf5 first.", file=sys.stderr)
        return 0

    train_loader, val_loader = get_dataloaders(
        train_h5, val_h5, batch_size=32, num_workers=0, val_seed=0
    )
    model = ChessMAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # 2 train steps
    model.train()
    it = iter(train_loader)
    for _ in range(2):
        enc, mask, target = next(it)
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

    # 1 val step
    model.eval()
    enc, mask, target = next(iter(val_loader))
    enc = enc.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast("cuda"):
                _, pred = model(enc, mask)
                loss = masked_mse_loss(pred, target, mask)
        else:
            _, pred = model(enc, mask)
            loss = masked_mse_loss(pred, target, mask)

    print(f"OK: ran 2 train + 1 val step on {device} (amp={use_amp})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
