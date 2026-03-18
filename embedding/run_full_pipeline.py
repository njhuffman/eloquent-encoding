#!/usr/bin/env python3
"""
Full pipeline: train MAE on train/val HDF5, compute test loss, run linear probes on a subset,
then write a report with training/val loss per epoch, final test loss, and probe train/val/test loss.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error
from torch.utils.data import DataLoader

from .config import (
    BATCH_SIZE,
    ELO_QUANTILE,
    LEARNING_RATE,
    NUM_EPOCHS,
    DATALOADER_NUM_WORKERS,
    PROBE_RANDOM_SEED,
    PROBE_SUBSET_RATIO,
    BOARD_HEIGHT,
    BOARD_WIDTH,
)
from .dataset import ChessBoardDataset, get_dataloaders
from .model import ChessMAE, masked_mse_loss


def _subsample_indices(n_total: int, ratio: float, seed: int) -> np.ndarray:
    n = max(1, int(round(ratio * n_total)))
    rng = np.random.default_rng(seed)
    return rng.choice(n_total, size=min(n, n_total), replace=False)


def extract_embeddings(
    model: ChessMAE,
    h5_path: Path,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    indices = np.asarray(indices)
    embeddings_list = []
    meta_list = []
    with h5py.File(h5_path, "r") as f:
        boards_ds = f["board"]
        meta_ds = f["meta"]
        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            batch_idx = indices[start:end]
            boards = np.stack([boards_ds[i] for i in batch_idx], axis=0).astype(np.float32)
            meta_batch = np.stack([meta_ds[i] for i in batch_idx], axis=0)
            enc_input = np.concatenate([
                boards,
                np.zeros((len(batch_idx), BOARD_HEIGHT, BOARD_WIDTH, 1), dtype=np.float32),
            ], axis=-1)
            with torch.no_grad():
                x = torch.from_numpy(enc_input).to(device)
                if x.shape[-1] == 19:
                    x = x.permute(0, 3, 1, 2)
                emb = model.encoder(x)
                emb = emb.cpu().numpy()
            embeddings_list.append(emb)
            meta_list.append(meta_batch)
    return np.vstack(embeddings_list), np.vstack(meta_list)


def run_mae_training(
    train_h5: Path,
    val_h5: Path,
    device: torch.device,
    epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    workers: int = DATALOADER_NUM_WORKERS,
    use_amp: bool = True,
    checkpoint_dir: Path | None = None,
) -> tuple[ChessMAE, list[float], list[float]]:
    """Train MAE; return (best model, train_loss_per_epoch, val_loss_per_epoch). Saves best checkpoint to checkpoint_dir."""
    train_loader, val_loader = get_dataloaders(
        train_h5, val_h5, batch_size=batch_size, num_workers=workers, val_seed=0
    )
    model = ChessMAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    best_val_loss = float("inf")
    best_state = None
    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for enc, mask, target in train_loader:
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
        train_loss /= max(n_batches, 1)
        train_losses.append(train_loss)

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
        val_losses.append(val_loss)

        if epoch == 1 and checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"epoch": 1, "model_state_dict": model.state_dict(), "val_loss": val_loss},
                checkpoint_dir / "epoch1.pt",
            )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_dir is not None:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": val_loss},
                    checkpoint_dir / "best.pt",
                )
        print(f"Epoch {epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}", file=sys.stderr)

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
    return model, train_losses, val_losses


def compute_mae_baseline_loss(
    test_h5: Path,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
    workers: int = 0,
    seed: int = 0,
    n_batches: int = 50,
) -> float:
    """
    Compute MSE of predicting the mean (per channel) on masked positions.
    Samples batches from test set; returns baseline loss (constant predictor).
    """
    test_ds = ChessBoardDataset(test_h5, seed=seed)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
    )
    # Collect targets and masks from n_batches
    targets_list = []
    masks_list = []
    for i, (_, mask, target) in enumerate(test_loader):
        if i >= n_batches:
            break
        # target/mask: (B, 8, 8, 12) and (B, 8, 8, 1)
        if target.shape[-1] == 12:
            target = target.permute(0, 3, 1, 2)  # (B, 12, 8, 8)
        if mask.shape[-1] == 1:
            mask = mask.permute(0, 3, 1, 2)  # (B, 1, 8, 8)
        targets_list.append(target)
        masks_list.append(mask)
    if not targets_list:
        return float("nan")
    targets = torch.cat(targets_list, dim=0)
    masks = torch.cat(masks_list, dim=0)
    # Mean per channel over masked positions only: for each of 12 channels, sum(target * mask) / sum(mask)
    # targets (N, 12, 8, 8), mask (N, 1, 8, 8) -> broadcast mask to 12 channels
    mask_expanded = masks.expand_as(targets)
    n_masked = mask_expanded.sum()
    if n_masked < 1:
        return float("nan")
    channel_mean = (targets * mask_expanded).sum(dim=(0, 2, 3)) / mask_expanded.sum(dim=(0, 2, 3))
    # Predict constant: (12,) broadcast to (N, 12, 8, 8)
    pred = channel_mean.view(1, 12, 1, 1).expand_as(targets)
    se = (pred - targets) ** 2
    masked_se = se * mask_expanded
    return (masked_se.sum() / n_masked).item()


def compute_mae_test_loss(
    model: ChessMAE,
    test_h5: Path,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
    workers: int = 0,
    use_amp: bool = True,
    seed: int = 0,
) -> float:
    test_ds = ChessBoardDataset(test_h5, seed=seed)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
    )
    model.eval()
    test_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for enc, mask, target in test_loader:
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
            test_loss += loss.item()
            n_batches += 1
    return test_loss / max(n_batches, 1)


def run_probes(
    model: ChessMAE,
    train_h5: Path,
    val_h5: Path,
    test_h5: Path,
    device: torch.device,
    subset_ratio: float = PROBE_SUBSET_RATIO,
    elo_quantile: float = ELO_QUANTILE,
    seed: int = PROBE_RANDOM_SEED,
) -> dict[str, dict[str, float]]:
    """
    Run all probes; return dict[probe_name, {"train_loss": float, "val_loss": float, "test_loss": float}].
    Loss is MSE for regression probes and log_loss for classification probes.
    """
    def get_len(p: Path) -> int:
        with h5py.File(p, "r") as f:
            return f["board"].shape[0]

    n_train = get_len(train_h5)
    n_val = get_len(val_h5)
    n_test = get_len(test_h5)
    train_idx = _subsample_indices(n_train, subset_ratio, seed)
    val_idx = _subsample_indices(n_val, subset_ratio, seed + 1)
    test_idx = _subsample_indices(n_test, subset_ratio, seed + 2)

    X_train, meta_train = extract_embeddings(model, train_h5, train_idx, device)
    X_val, meta_val = extract_embeddings(model, val_h5, val_idx, device)
    X_test, meta_test = extract_embeddings(model, test_h5, test_idx, device)
    # meta: elo_white, elo_black, piece_count_white, piece_count_black, outcome, in_check

    results = {}

    def _regression_baseline(y_train: np.ndarray, y_test: np.ndarray) -> tuple[float, float]:
        pred_const = np.full_like(y_test, y_train.mean(), dtype=np.float64)
        return float(mean_squared_error(y_test, pred_const)), float(y_train.mean())

    def _classification_baseline(y_train: np.ndarray, y_test: np.ndarray) -> tuple[float, float]:
        p = float(y_train.mean())
        p = max(1e-15, min(1 - 1e-15, p))
        return float(log_loss(y_test, np.full(len(y_test), p))), p

    # 1. Piece count (regression) - MSE as loss
    for probe_name, col in [("piece_count_white", 2), ("piece_count_black", 3)]:
        y_train = meta_train[:, col]
        y_val = meta_val[:, col]
        y_test = meta_test[:, col]
        reg = Ridge(alpha=1.0, random_state=seed).fit(X_train, y_train)
        test_loss = float(mean_squared_error(y_test, reg.predict(X_test)))
        baseline_test, _ = _regression_baseline(y_train, y_test)
        improvement = (1 - test_loss / baseline_test) * 100 if baseline_test > 0 else float("nan")
        results[probe_name] = {
            "train_loss": float(mean_squared_error(y_train, reg.predict(X_train))),
            "val_loss": float(mean_squared_error(y_val, reg.predict(X_val))),
            "test_loss": test_loss,
            "baseline_test": baseline_test,
            "improvement_pct": improvement,
        }

    # 2. in_check (classification) - log_loss
    y_train = (meta_train[:, 5] >= 0.5).astype(int)
    y_val = (meta_val[:, 5] >= 0.5).astype(int)
    y_test = (meta_test[:, 5] >= 0.5).astype(int)
    clf = LogisticRegression(max_iter=1000, random_state=seed).fit(X_train, y_train)
    proba_train = clf.predict_proba(X_train)[:, 1]
    proba_val = clf.predict_proba(X_val)[:, 1]
    proba_test = clf.predict_proba(X_test)[:, 1]
    test_loss = float(log_loss(y_test, proba_test))
    baseline_test, _ = _classification_baseline(y_train, y_test)
    improvement = (1 - test_loss / baseline_test) * 100 if baseline_test > 0 else float("nan")
    results["in_check"] = {
        "train_loss": float(log_loss(y_train, proba_train)),
        "val_loss": float(log_loss(y_val, proba_val)),
        "test_loss": test_loss,
        "baseline_test": baseline_test,
        "improvement_pct": improvement,
    }

    # 3. Elo regression - MSE
    mean_elo_train = (meta_train[:, 0] + meta_train[:, 1]) / 2
    mean_elo_val = (meta_val[:, 0] + meta_val[:, 1]) / 2
    mean_elo_test = (meta_test[:, 0] + meta_test[:, 1]) / 2
    reg = Ridge(alpha=1.0, random_state=seed).fit(X_train, mean_elo_train)
    test_loss = float(mean_squared_error(mean_elo_test, reg.predict(X_test)))
    baseline_test, _ = _regression_baseline(mean_elo_train, mean_elo_test)
    improvement = (1 - test_loss / baseline_test) * 100 if baseline_test > 0 else float("nan")
    results["elo_regression"] = {
        "train_loss": float(mean_squared_error(mean_elo_train, reg.predict(X_train))),
        "val_loss": float(mean_squared_error(mean_elo_val, reg.predict(X_val))),
        "test_loss": test_loss,
        "baseline_test": baseline_test,
        "improvement_pct": improvement,
    }

    # 4. Elo top vs bottom - log_loss
    q_lo = elo_quantile
    q_hi = 1.0 - elo_quantile
    thresh_lo = np.quantile(mean_elo_train, q_lo)
    thresh_hi = np.quantile(mean_elo_train, q_hi)
    train_mask = (mean_elo_train <= thresh_lo) | (mean_elo_train >= thresh_hi)
    y_train_bin = (mean_elo_train >= thresh_hi).astype(int)
    X_train_bin = X_train[train_mask]
    y_train_bin = y_train_bin[train_mask]
    y_val_bin = (mean_elo_val >= thresh_hi).astype(int)
    y_test_bin = (mean_elo_test >= thresh_hi).astype(int)
    clf = LogisticRegression(max_iter=1000, random_state=seed).fit(X_train_bin, y_train_bin)
    proba_val = clf.predict_proba(X_val)[:, 1]
    proba_test = clf.predict_proba(X_test)[:, 1]
    test_loss = float(log_loss(y_test_bin, proba_test))
    baseline_test, _ = _classification_baseline(y_train_bin, y_test_bin)
    improvement = (1 - test_loss / baseline_test) * 100 if baseline_test > 0 else float("nan")
    results["elo_top_vs_bottom"] = {
        "train_loss": float(log_loss(y_train_bin, clf.predict_proba(X_train_bin)[:, 1])),
        "val_loss": float(log_loss(y_val_bin, proba_val)),
        "test_loss": test_loss,
        "baseline_test": baseline_test,
        "improvement_pct": improvement,
    }

    return results


def _write_report_figures(
    report_path: Path,
    mae_train_losses: list[float],
    mae_val_losses: list[float],
    probe_results_epoch1: dict[str, dict[str, float]] | None,
    probe_results_final: dict[str, dict[str, float]],
) -> dict[str, str]:
    """
    Write figure PNGs to report_path stem_figs/; return dict of figure name -> relative path for markdown.
    If matplotlib is not available, return empty dict and skip plots.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {}
    fig_dir = report_path.parent / f"{report_path.stem}_figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rel = f"{report_path.stem}_figs"
    paths = {}

    # MAE loss curve
    fig, ax = plt.subplots()
    ax.plot(range(1, len(mae_train_losses) + 1), mae_train_losses, label="Train loss")
    ax.plot(range(1, len(mae_val_losses) + 1), mae_val_losses, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE on masked positions)")
    ax.set_title("MAE training and validation loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    p = fig_dir / "mae_loss_curve.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths["mae_curve"] = f"{rel}/mae_loss_curve.png"

    # Probe test loss: epoch 1 vs final (grouped bar)
    if probe_results_epoch1 is not None and probe_results_final:
        probes = list(probe_results_final.keys())
        x = np.arange(len(probes))
        w = 0.35
        test_e1 = [probe_results_epoch1[p]["test_loss"] for p in probes]
        test_fin = [probe_results_final[p]["test_loss"] for p in probes]
        fig, ax = plt.subplots()
        ax.bar(x - w / 2, test_e1, w, label="Test loss (epoch 1)")
        ax.bar(x + w / 2, test_fin, w, label="Test loss (final)")
        ax.set_xticks(x)
        ax.set_xticklabels(probes, rotation=15, ha="right")
        ax.set_ylabel("Test loss")
        ax.set_title("Probe test loss: epoch 1 vs final")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        p = fig_dir / "probe_progress.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths["probe_progress"] = f"{rel}/probe_progress.png"

    return paths


def write_report(
    report_path: Path,
    mae_train_losses: list[float],
    mae_val_losses: list[float],
    mae_test_loss: float,
    mae_baseline_loss: float,
    mae_improvement_pct: float,
    probe_results_epoch1: dict[str, dict[str, float]] | None,
    probe_results_final: dict[str, dict[str, float]],
) -> None:
    fig_paths = _write_report_figures(
        report_path, mae_train_losses, mae_val_losses, probe_results_epoch1, probe_results_final
    )

    with open(report_path, "w") as f:
        f.write("# Embedding pipeline report\n\n")
        f.write("## How to read this report\n\n")
        f.write("- **MAE loss**: Mean squared error (MSE) between predicted and true 8×8×12 piece planes, ")
        f.write("averaged only over **masked** squares. Unit: squared error per masked position per channel; ")
        f.write("targets are 0/1 so scale is 0–1. Lower is better; typical range 0.01–0.05 after training.\n\n")
        f.write("- **Probe losses**: Regression probes (piece count, elo) report **MSE** (piece count in count², ")
        f.write("elo in Elo²). Classification probes (in_check, elo top vs bottom) report **log loss** (nats; ")
        f.write("random guessing ≈ 0.69). Lower is better.\n\n")
        f.write("- **Baseline**: Loss of predicting the **mean** (constant predictor). ")
        f.write("**Improvement %** = (1 − model_loss / baseline_loss) × 100; higher means the model is better than guessing the average.\n\n")

        f.write("## 1. MAE training\n\n")
        f.write("| Epoch | Train loss | Val loss |\n")
        f.write("|-------|------------|----------|\n")
        for ep, (tl, vl) in enumerate(zip(mae_train_losses, mae_val_losses), start=1):
            f.write(f"| {ep} | {tl:.6f} | {vl:.6f} |\n")
        f.write("\n")
        f.write("- **MAE baseline (predict mean):** {:.6f}\n".format(mae_baseline_loss))
        f.write("- **Improvement over baseline:** {:.1f}%\n".format(mae_improvement_pct))
        f.write("- **Final test loss (MAE):** {:.6f}\n\n".format(mae_test_loss))
        if "mae_curve" in fig_paths:
            f.write("![MAE loss curve]({})\n\n".format(fig_paths["mae_curve"]))

        f.write("## 2. Linear probes (subset)\n\n")
        has_epoch1 = probe_results_epoch1 is not None
        if has_epoch1:
            f.write("| Probe | Train loss (final) | Val loss (final) | Test loss (epoch 1) | Test loss (final) | Baseline (test) | Improvement % |\n")
            f.write("|-------|-------------------|------------------|---------------------|-------------------|-----------------|----------------|\n")
            for name, m in probe_results_final.items():
                te1 = probe_results_epoch1[name]["test_loss"]
                f.write(
                    "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.1f}% |\n".format(
                        name,
                        m["train_loss"],
                        m["val_loss"],
                        te1,
                        m["test_loss"],
                        m["baseline_test"],
                        m["improvement_pct"],
                    )
                )
        else:
            f.write("| Probe | Train loss | Val loss | Test loss | Baseline (test) | Improvement % |\n")
            f.write("|-------|------------|----------|----------|-----------------|----------------|\n")
            for name, m in probe_results_final.items():
                f.write(
                    "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.1f}% |\n".format(
                        name,
                        m["train_loss"],
                        m["val_loss"],
                        m["test_loss"],
                        m["baseline_test"],
                        m["improvement_pct"],
                    )
                )
        f.write("\n")
        if "probe_progress" in fig_paths:
            f.write("![Probe test loss: epoch 1 vs final]({})\n\n".format(fig_paths["probe_progress"]))

        f.write("## Summary\n\n")
        f.write("MAE is **{:.1f}%** better than baseline (predict mean). ".format(mae_improvement_pct))
        improvements = [m["improvement_pct"] for m in probe_results_final.values()]
        f.write("Probe improvements over baseline (test): " + ", ".join(f"{x:.1f}%" for x in improvements) + ".\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Full pipeline: train MAE, run probes, write report."
    )
    parser.add_argument("--train-h5", type=Path, required=True, help="Train HDF5")
    parser.add_argument("--val-h5", type=Path, required=True, help="Val HDF5")
    parser.add_argument("--test-h5", type=Path, required=True, help="Test HDF5")
    parser.add_argument("--report", type=Path, default=Path("embedding_report.md"), help="Output report path")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="MAE epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--workers", type=int, default=DATALOADER_NUM_WORKERS, help="DataLoader workers")
    parser.add_argument("--subset-ratio", type=float, default=PROBE_SUBSET_RATIO, help="Probe data subset ratio")
    parser.add_argument("--elo-quantile", type=float, default=ELO_QUANTILE, help="Elo top/bottom quantile for binary probe")
    parser.add_argument("--seed", type=int, default=PROBE_RANDOM_SEED, help="Random seed for probes")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"), help="Where to save best MAE checkpoint")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    args = parser.parse_args()

    for p in (args.train_h5, args.val_h5, args.test_h5):
        if not p.exists():
            print(f"Error: not found {p}", file=sys.stderr)
            return 1

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Error: --device cuda requested but CUDA not available.", file=sys.stderr)
        return 1
    use_amp = not args.no_amp and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}, AMP: {use_amp}", file=sys.stderr)

    print("Training MAE...", file=sys.stderr)
    model, train_losses, val_losses = run_mae_training(
        args.train_h5,
        args.val_h5,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        workers=args.workers,
        use_amp=use_amp,
        checkpoint_dir=args.checkpoint_dir,
    )

    print("Computing MAE test loss...", file=sys.stderr)
    mae_test_loss = compute_mae_test_loss(
        model, args.test_h5, device, batch_size=args.batch_size, workers=args.workers, use_amp=use_amp, seed=args.seed
    )
    print(f"MAE test loss: {mae_test_loss:.6f}", file=sys.stderr)

    print("Computing MAE baseline (predict mean)...", file=sys.stderr)
    mae_baseline_loss = compute_mae_baseline_loss(
        args.test_h5, device, batch_size=args.batch_size, workers=args.workers, seed=args.seed
    )
    mae_improvement_pct = (1 - mae_test_loss / mae_baseline_loss) * 100 if mae_baseline_loss > 0 else float("nan")
    print(f"MAE baseline: {mae_baseline_loss:.6f}, improvement: {mae_improvement_pct:.1f}%", file=sys.stderr)

    probe_results_epoch1 = None
    epoch1_ckpt = args.checkpoint_dir / "epoch1.pt"
    if epoch1_ckpt.exists():
        print("Running probes (epoch 1 checkpoint)...", file=sys.stderr)
        model_epoch1 = ChessMAE()
        state = torch.load(epoch1_ckpt, map_location=device, weights_only=False)
        model_epoch1.load_state_dict(state["model_state_dict"])
        model_epoch1 = model_epoch1.to(device)
        probe_results_epoch1 = run_probes(
            model_epoch1,
            args.train_h5,
            args.val_h5,
            args.test_h5,
            device,
            subset_ratio=args.subset_ratio,
            elo_quantile=args.elo_quantile,
            seed=args.seed,
        )

    print("Running probes (final model)...", file=sys.stderr)
    probe_results_final = run_probes(
        model,
        args.train_h5,
        args.val_h5,
        args.test_h5,
        device,
        subset_ratio=args.subset_ratio,
        elo_quantile=args.elo_quantile,
        seed=args.seed,
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_report(
        args.report,
        mae_train_losses=train_losses,
        mae_val_losses=val_losses,
        mae_test_loss=mae_test_loss,
        mae_baseline_loss=mae_baseline_loss,
        mae_improvement_pct=mae_improvement_pct,
        probe_results_epoch1=probe_results_epoch1,
        probe_results_final=probe_results_final,
    )
    print(f"Report written to {args.report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
