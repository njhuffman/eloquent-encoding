#!/usr/bin/env python3
"""Train move predictor (3-way cross-entropy) from move HDF5."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from move_predictor.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DATALOADER_NUM_WORKERS,
    DROPOUT,
    GRU_HIDDEN,
    GRU_NUM_LAYERS,
    LEARNING_RATE,
    MLP_HIDDEN,
    MOVE_EMB_DIM,
    NUM_EPOCHS,
    TURN_EMB_DIM,
)
from move_predictor.dataset import MovePredictorH5Dataset
from move_predictor.model import MovePredictor


def _load_h5_attrs(path: Path) -> tuple[int, int]:
    import h5py

    with h5py.File(path, "r") as f:
        return int(f.attrs["embedding_dim"]), int(f.attrs["history_n"])


def _to_device(
    batch: tuple[torch.Tensor, ...],
    device: torch.device,
    non_blocking: bool,
) -> tuple[torch.Tensor, ...]:
    cur, hw, hb, lw, lb, turn, fs, ts, y = batch
    return (
        cur.to(device, non_blocking=non_blocking),
        hw.to(device, non_blocking=non_blocking),
        hb.to(device, non_blocking=non_blocking),
        lw.to(device, non_blocking=non_blocking),
        lb.to(device, non_blocking=non_blocking),
        turn.to(device, non_blocking=non_blocking),
        fs.to(device, non_blocking=non_blocking),
        ts.to(device, non_blocking=non_blocking),
        y.to(device, non_blocking=non_blocking),
    )


def _make_loader(
    path: Path,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    pin_memory: bool,
    drop_last: bool,
) -> DataLoader:
    ds = MovePredictorH5Dataset(path)
    base_sampler = RandomSampler(ds) if shuffle else SequentialSampler(ds)
    batch_sampler = BatchSampler(base_sampler, batch_size, drop_last=drop_last)
    kw: dict = dict(
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        kw["prefetch_factor"] = 2
        kw["persistent_workers"] = True
    return DataLoader(ds, **kw)


def _log_epoch_pipeline(
    *,
    epoch: int,
    n_batches: int,
    wait_loader_s: float,
    h2d_s: float,
    gpu_s: float,
    sync_gpu: bool,
    file=sys.stderr,
) -> None:
    """Mean times over train batches (batch 0 skipped — cold start)."""
    n = max(n_batches - 1, 1)
    wl = wait_loader_s / n
    hd = h2d_s / n
    gp = gpu_s / n
    tot = wl + hd + gp
    if tot <= 0:
        return
    print(
        f"[pipeline] epoch {epoch + 1}  train batches={n_batches}  "
        f"per batch (mean, skip batch0):  "
        f"wait_dataloader={wl * 1e3:.2f}ms ({100.0 * wl / tot:.0f}%)  "
        f"host_to_device={hd * 1e3:.2f}ms ({100.0 * hd / tot:.0f}%)  "
        f"forward+backward+step={gp * 1e3:.2f}ms ({100.0 * gp / tot:.0f}%)",
        file=file,
    )
    if sync_gpu:
        print(
            "[pipeline] GPU times used torch.cuda.synchronize() each batch (true device wait).",
            file=file,
        )
    else:
        print(
            "[pipeline] GPU segment is wall time after launch (often overlaps H2D); "
            "use --profile-sync-gpu for synchronized GPU time.",
            file=file,
        )
    if wl / tot > 0.25:
        print(
            "[pipeline] dataloader share is high — try more --workers, faster disk, or larger --batch-size.",
            file=file,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train move predictor")
    parser.add_argument("--train-h5", type=Path, required=True)
    parser.add_argument("--val-h5", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--workers", type=int, default=DATALOADER_NUM_WORKERS)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--move-emb-dim", type=int, default=MOVE_EMB_DIM)
    parser.add_argument("--turn-emb-dim", type=int, default=TURN_EMB_DIM)
    parser.add_argument("--gru-hidden", type=int, default=GRU_HIDDEN)
    parser.add_argument("--mlp-hidden", type=int, default=MLP_HIDDEN)
    parser.add_argument(
        "--no-pipeline-log",
        action="store_true",
        help="Disable per-epoch [pipeline] lines (dataloader vs H2D vs GPU timing).",
    )
    parser.add_argument(
        "--profile-sync-gpu",
        action="store_true",
        help="torch.cuda.synchronize() after each train batch so GPU segment reflects device wait (slower).",
    )
    args = parser.parse_args()

    try:
        import torch.multiprocessing as mp

        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass

    if not args.train_h5.is_file() or not args.val_h5.is_file():
        print("Error: train-h5 or val-h5 missing", file=sys.stderr)
        return 1

    emb_d, hist_n = _load_h5_attrs(args.train_h5)
    val_emb, val_hist = _load_h5_attrs(args.val_h5)
    if emb_d != val_emb or hist_n != val_hist:
        print("Error: train and val HDF5 attrs mismatch", file=sys.stderr)
        return 1

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.checkpoint_dir / "best.pt"
    non_blocking = device.type == "cuda"
    cuda = device.type == "cuda"

    train_loader = _make_loader(
        args.train_h5,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=cuda,
        drop_last=True,
    )
    val_loader = _make_loader(
        args.val_h5,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=cuda,
        drop_last=False,
    )

    model = MovePredictor(
        embedding_dim=emb_d,
        history_n=hist_n,
        move_emb_dim=args.move_emb_dim,
        turn_emb_dim=args.turn_emb_dim,
        gru_hidden=args.gru_hidden,
        gru_num_layers=GRU_NUM_LAYERS,
        mlp_hidden=args.mlp_hidden,
        dropout=DROPOUT,
    ).to(device)

    print(
        f"Initializing MovePredictor with embedding_dim={emb_d}, history_n={hist_n}, "
        f"move_emb_dim={args.move_emb_dim}, turn_emb_dim={args.turn_emb_dim}, "
        f"gru_hidden={args.gru_hidden}, "
        f"gru_num_layers={GRU_NUM_LAYERS}, mlp_hidden={args.mlp_hidden}, dropout={DROPOUT}",
        file=sys.stderr,
    )
    print(
        f"[train] HDF5: BatchSampler + slice reads (__getitems__), "
        f"workers={args.workers}, batch_size={args.batch_size}",
        file=sys.stderr,
    )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    next_tick = time.perf_counter()
    pipeline_log = not args.no_pipeline_log

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        n = 0
        correct = 0
        sum_wait_loader = 0.0
        sum_h2d = 0.0
        sum_gpu = 0.0
        n_train_batches = 0
        for bi, batch in enumerate(train_loader):
            t_after_iter = time.perf_counter()
            wait_loader = t_after_iter - next_tick

            t_before_h2d = time.perf_counter()
            cur, hw, hb, lw, lb, turn, fs, ts, y = _to_device(batch, device, non_blocking)
            t_after_h2d = time.perf_counter()
            h2d = t_after_h2d - t_before_h2d

            t_gpu_start = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            logits = model(cur, hw, hb, lw, lb, turn, fs, ts)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            if args.profile_sync_gpu and cuda:
                torch.cuda.synchronize()
            next_tick = time.perf_counter()
            gpu_seg = next_tick - t_gpu_start

            if pipeline_log and bi > 0:
                sum_wait_loader += wait_loader
                sum_h2d += h2d
                sum_gpu += gpu_seg

            total += loss.item() * y.size(0)
            n += y.size(0)
            correct += (logits.argmax(dim=-1) == y).sum().item()
            n_train_batches += 1

        if pipeline_log and n_train_batches > 1:
            _log_epoch_pipeline(
                epoch=epoch,
                n_batches=n_train_batches,
                wait_loader_s=sum_wait_loader,
                h2d_s=sum_h2d,
                gpu_s=sum_gpu,
                sync_gpu=bool(args.profile_sync_gpu and cuda),
            )
        elif pipeline_log and n_train_batches <= 1:
            print(
                "[pipeline] skipped (need at least 2 train batches for mean after batch0)",
                file=sys.stderr,
            )

        train_loss = total / max(n, 1)
        train_acc = correct / max(n, 1)

        model.eval()
        vtotal = 0.0
        vn = 0
        vcorrect = 0
        with torch.no_grad():
            for batch in val_loader:
                cur, hw, hb, lw, lb, turn, fs, ts, y = _to_device(batch, device, non_blocking)
                logits = model(cur, hw, hb, lw, lb, turn, fs, ts)
                loss = loss_fn(logits, y)
                vtotal += loss.item() * y.size(0)
                vn += y.size(0)
                vcorrect += (logits.argmax(dim=-1) == y).sum().item()
        val_loss = vtotal / max(vn, 1)
        val_acc = vcorrect / max(vn, 1)
        print(
            f"epoch {epoch + 1}/{args.epochs}  train_loss={train_loss:.4f} acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f}",
            file=sys.stderr,
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "embedding_dim": emb_d,
                    "history_n": hist_n,
                    "move_emb_dim": args.move_emb_dim,
                    "turn_emb_dim": args.turn_emb_dim,
                    "gru_hidden": args.gru_hidden,
                    "mlp_hidden": args.mlp_hidden,
                    "gru_num_layers": GRU_NUM_LAYERS,
                    "dropout": DROPOUT,
                },
                best_path,
            )
            print(f"  saved {best_path}", file=sys.stderr)

    print(f"Best val loss: {best_val:.4f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
