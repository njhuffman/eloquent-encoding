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
import torch.nn as nn
from sklearn.metrics import log_loss, mean_squared_error
from torch.utils.data import DataLoader

from .architectures import DEFAULT_ARCHITECTURE_ID, build_model, resolve_config_for_id
from .checkpoint_utils import build_model_checkpoint
from .config import (
    ARTIFACTS_DIR,
    BATCH_SIZE,
    ELO_QUANTILE,
    LEARNING_RATE,
    NUM_EPOCHS,
    DATALOADER_NUM_WORKERS,
    PROBE_EPOCHS,
    PROBE_LR,
    PROBE_RANDOM_SEED,
    PROBE_SUBSET_RATIO,
    BOARD_HEIGHT,
    BOARD_WIDTH,
)
from .dataset import ChessBoardDataset, get_dataloaders
from .model import masked_mse_loss
from .registry import generate_model_name, parse_arch_config_arg, register_model
from .probes import (
    predict_classifier_proba,
    predict_regression,
    train_classifier_probe,
    train_regression_probe,
)


class _CUDAPrefetcher:
    """Prefetches the next batch to GPU on a separate stream so transfer overlaps with compute."""

    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None

    def __iter__(self):
        self.iter = iter(self.loader)
        self._preload()
        return self

    def _preload(self):
        try:
            batch = next(self.iter)
        except StopIteration:
            self._next = None
            return
        if self.stream is None:
            self._next = (
                batch[0].to(self.device, non_blocking=True),
                batch[1].to(self.device, non_blocking=True),
                batch[2].to(self.device, non_blocking=True),
            )
            return
        with torch.cuda.stream(self.stream):
            self._next = (
                batch[0].to(self.device, non_blocking=True),
                batch[1].to(self.device, non_blocking=True),
                batch[2].to(self.device, non_blocking=True),
            )

    def __next__(self):
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        if self._next is None:
            raise StopIteration
        out = self._next
        self._preload()
        return out


def _subsample_indices(n_total: int, ratio: float, seed: int) -> np.ndarray:
    n = max(1, int(round(ratio * n_total)))
    rng = np.random.default_rng(seed)
    return rng.choice(n_total, size=min(n, n_total), replace=False)


def extract_embeddings(
    model: nn.Module,
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


def extract_raw_inputs(
    h5_path: Path,
    indices: np.ndarray,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load boards at indices, build encoder input (8,8,19), flatten to (n, 8*8*19).
    Return (X, meta) with X shape (len(indices), 1216).
    """
    indices = np.asarray(indices)
    X_list = []
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
            # Flatten (B, 8, 8, 19) -> (B, 1216)
            X_list.append(enc_input.reshape(len(batch_idx), -1))
            meta_list.append(meta_batch)
    return np.vstack(X_list).astype(np.float32), np.vstack(meta_list)


def run_probes_on_embeddings(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    meta_train: np.ndarray,
    meta_val: np.ndarray,
    meta_test: np.ndarray,
    device: torch.device,
    seed: int = PROBE_RANDOM_SEED,
    elo_quantile: float = ELO_QUANTILE,
) -> dict[str, dict[str, float]]:
    """
    Run all probes on precomputed (X_train, X_val, X_test) and meta arrays.
    Return same dict as run_probes: probe_name -> {train_loss, val_loss, test_loss, baseline_test, improvement_pct}.
    """
    results = {}

    def _regression_baseline(y_train: np.ndarray, y_test: np.ndarray) -> tuple[float, float]:
        pred_const = np.full_like(y_test, y_train.mean(), dtype=np.float64)
        return float(mean_squared_error(y_test, pred_const)), float(y_train.mean())

    def _classification_baseline(y_train: np.ndarray, y_test: np.ndarray) -> tuple[float, float]:
        p = float(y_train.mean())
        p = max(1e-15, min(1 - 1e-15, p))
        return float(log_loss(y_test, np.full(len(y_test), p))), p

    # 1. Piece count (regression)
    for probe_name, col in [("piece_count_white", 2), ("piece_count_black", 3)]:
        y_train = meta_train[:, col]
        y_val = meta_val[:, col]
        y_test = meta_test[:, col]
        probe = train_regression_probe(
            X_train, y_train, device, seed=seed, epochs=PROBE_EPOCHS, lr=PROBE_LR
        )
        pred_train = predict_regression(probe, X_train, device)
        pred_val = predict_regression(probe, X_val, device)
        pred_test = predict_regression(probe, X_test, device)
        test_loss = float(mean_squared_error(y_test, pred_test))
        baseline_test, _ = _regression_baseline(y_train, y_test)
        improvement = (1 - test_loss / baseline_test) * 100 if baseline_test > 0 else float("nan")
        results[probe_name] = {
            "train_loss": float(mean_squared_error(y_train, pred_train)),
            "val_loss": float(mean_squared_error(y_val, pred_val)),
            "test_loss": test_loss,
            "baseline_test": baseline_test,
            "improvement_pct": improvement,
        }

    # 2. in_check (classification)
    y_train = (meta_train[:, 5] >= 0.5).astype(int)
    y_val = (meta_val[:, 5] >= 0.5).astype(int)
    y_test = (meta_test[:, 5] >= 0.5).astype(int)
    probe = train_classifier_probe(
        X_train, y_train, device, seed=seed, epochs=PROBE_EPOCHS, lr=PROBE_LR
    )
    proba_train = predict_classifier_proba(probe, X_train, device)
    proba_val = predict_classifier_proba(probe, X_val, device)
    proba_test = predict_classifier_proba(probe, X_test, device)
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

    # 3. Elo regression
    mean_elo_train = (meta_train[:, 0] + meta_train[:, 1]) / 2
    mean_elo_val = (meta_val[:, 0] + meta_val[:, 1]) / 2
    mean_elo_test = (meta_test[:, 0] + meta_test[:, 1]) / 2
    probe = train_regression_probe(
        X_train, mean_elo_train, device, seed=seed, epochs=PROBE_EPOCHS, lr=PROBE_LR
    )
    pred_train = predict_regression(probe, X_train, device)
    pred_val = predict_regression(probe, X_val, device)
    pred_test = predict_regression(probe, X_test, device)
    test_loss = float(mean_squared_error(mean_elo_test, pred_test))
    baseline_test, _ = _regression_baseline(mean_elo_train, mean_elo_test)
    improvement = (1 - test_loss / baseline_test) * 100 if baseline_test > 0 else float("nan")
    results["elo_regression"] = {
        "train_loss": float(mean_squared_error(mean_elo_train, pred_train)),
        "val_loss": float(mean_squared_error(mean_elo_val, pred_val)),
        "test_loss": test_loss,
        "baseline_test": baseline_test,
        "improvement_pct": improvement,
    }

    # 4. Elo top vs bottom
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
    probe = train_classifier_probe(
        X_train_bin, y_train_bin, device, seed=seed, epochs=PROBE_EPOCHS, lr=PROBE_LR
    )
    proba_train_bin = predict_classifier_proba(probe, X_train_bin, device)
    proba_val = predict_classifier_proba(probe, X_val, device)
    proba_test = predict_classifier_proba(probe, X_test, device)
    test_loss = float(log_loss(y_test_bin, proba_test))
    baseline_test, _ = _classification_baseline(y_train_bin, y_test_bin)
    improvement = (1 - test_loss / baseline_test) * 100 if baseline_test > 0 else float("nan")
    results["elo_top_vs_bottom"] = {
        "train_loss": float(log_loss(y_train_bin, proba_train_bin)),
        "val_loss": float(log_loss(y_val_bin, proba_val)),
        "test_loss": test_loss,
        "baseline_test": baseline_test,
        "improvement_pct": improvement,
    }

    return results


def _h5_board_counts(train_h5: Path, val_h5: Path) -> tuple[int, int]:
    with h5py.File(train_h5, "r") as f:
        n_train = f["board"].shape[0]
    with h5py.File(val_h5, "r") as f:
        n_val = f["board"].shape[0]
    return n_train, n_val


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
    in_memory: bool = True,
    use_compile: bool = False,
    profile: bool = False,
    architecture_id: str = DEFAULT_ARCHITECTURE_ID,
    architecture_config: dict | None = None,
) -> tuple[nn.Module, list[float], list[float], dict]:
    """Train MAE; return (best model, train_losses, val_losses, info dict). Saves best checkpoint to checkpoint_dir."""
    n_train, n_val = _h5_board_counts(train_h5, val_h5)
    train_meta = {
        "n_train_boards": int(n_train),
        "n_val_boards": int(n_val),
        "train_h5_basename": train_h5.name,
        "val_h5_basename": val_h5.name,
    }
    resolved_architecture_config = resolve_config_for_id(architecture_id, architecture_config)

    train_loader, val_loader = get_dataloaders(
        train_h5,
        val_h5,
        batch_size=batch_size,
        num_workers=workers,
        val_seed=0,
        in_memory=in_memory,
    )
    model = build_model(architecture_id, architecture_config).to(device)
    if use_compile:
        model = torch.compile(model, mode="reduce-overhead")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        prof = (
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=5, active=20, repeat=0),
                on_trace_ready=None,
                record_shapes=False,
                profile_memory=False,
            )
            if profile and epoch == 1
            else None
        )
        if prof is not None:
            prof.start()
        train_iter = _CUDAPrefetcher(train_loader, device) if device.type == "cuda" else train_loader
        for batch_idx, (enc, mask, target) in enumerate(train_iter):
            if prof is not None:
                prof.step()
            if device.type != "cuda":
                enc = enc.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
            # On CUDA, prefetcher already moved batch to device
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
        if prof is not None:
            prof.stop()
            print(prof.key_averages().table(sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total", row_limit=15), file=sys.stderr)
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_dir is not None:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                payload = build_model_checkpoint(
                    model,
                    architecture_id=architecture_id,
                    architecture_config=resolved_architecture_config,
                    train_meta=train_meta,
                    train_hparams={
                        "batch_size": batch_size,
                        "lr": lr,
                        "epochs_requested": epochs,
                        "epoch": epoch,
                        "best_val_loss": val_loss,
                    },
                    epoch=epoch,
                    val_loss=val_loss,
                )
                torch.save(payload, checkpoint_dir / "best.pt")
        print(f"Epoch {epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}", file=sys.stderr)

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
    info = {
        "architecture_id": architecture_id,
        "architecture_config": resolved_architecture_config,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss if best_state is not None else float("nan"),
        "train_meta": train_meta,
    }
    return model, train_losses, val_losses, info


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
    model: nn.Module,
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
    model: nn.Module,
    train_h5: Path,
    val_h5: Path,
    test_h5: Path,
    device: torch.device,
    subset_ratio: float = PROBE_SUBSET_RATIO,
    elo_quantile: float = ELO_QUANTILE,
    seed: int = PROBE_RANDOM_SEED,
) -> dict[str, dict[str, float]]:
    """
    Run all probes on embeddings from the given model.
    Return dict[probe_name, {train_loss, val_loss, test_loss, baseline_test, improvement_pct}].
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
    return run_probes_on_embeddings(
        X_train, X_val, X_test, meta_train, meta_val, meta_test, device, seed=seed, elo_quantile=elo_quantile
    )


def _write_report_figures(
    report_path: Path,
    mae_train_losses: list[float],
    mae_val_losses: list[float],
    probe_results_final: dict[str, dict[str, float]],
    probe_results_random_emb: dict[str, dict[str, float]],
    probe_results_random_model: dict[str, dict[str, float]],
    probe_results_raw_input: dict[str, dict[str, float]],
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

    # Probe test loss: one subplot per probe so different scales are all visible
    if probe_results_final:
        probes = list(probe_results_final.keys())
        n_probes = len(probes)
        fig, axes = plt.subplots(1, n_probes, figsize=(4 * n_probes, 4), squeeze=False)
        axes = axes[0]
        labels = ["Final model", "Random emb", "Random model", "Raw input"]
        x = np.arange(4)
        width = 0.65
        for i, probe_name in enumerate(probes):
            ax = axes[i]
            vals = [
                probe_results_final[probe_name]["test_loss"],
                probe_results_random_emb[probe_name]["test_loss"],
                probe_results_random_model[probe_name]["test_loss"],
                probe_results_raw_input[probe_name]["test_loss"],
            ]
            bars = ax.bar(x, vals, width=width)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha="right")
            ax.set_ylabel("Test loss")
            ax.set_title(probe_name)
            ax.grid(True, alpha=0.3, axis="y")
        fig.suptitle("Probe test loss: final model vs baselines (one y-scale per probe)", fontsize=10)
        fig.tight_layout()
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
    probe_results_final: dict[str, dict[str, float]],
    probe_results_random_emb: dict[str, dict[str, float]],
    probe_results_random_model: dict[str, dict[str, float]],
    probe_results_raw_input: dict[str, dict[str, float]],
) -> None:
    fig_paths = _write_report_figures(
        report_path,
        mae_train_losses,
        mae_val_losses,
        probe_results_final,
        probe_results_random_emb,
        probe_results_random_model,
        probe_results_raw_input,
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
        f.write("Test loss comparison: **final (trained) model** vs **random embedding**, **random-weights model**, **raw input** (flattened 8×8×19).\n\n")
        f.write("| Probe | Final (test) | Random emb | Random model | Raw input | Baseline (test) | Improvement % (final) |\n")
        f.write("|-------|--------------|------------|--------------|----------|-----------------|------------------------|\n")
        for name in probe_results_final:
            m = probe_results_final[name]
            r_emb = probe_results_random_emb[name]["test_loss"]
            r_mod = probe_results_random_model[name]["test_loss"]
            raw = probe_results_raw_input[name]["test_loss"]
            f.write(
                "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.1f}% |\n".format(
                    name,
                    m["test_loss"],
                    r_emb,
                    r_mod,
                    raw,
                    m["baseline_test"],
                    m["improvement_pct"],
                )
            )
        f.write("\n")
        if "probe_progress" in fig_paths:
            f.write("![Probe test loss: final vs baselines]({})\n\n".format(fig_paths["probe_progress"]))

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
    parser.add_argument(
    "--workers",
    type=int,
    default=DATALOADER_NUM_WORKERS,
    help="DataLoader workers (use 0 or 2 if you hit 'Bus error' / out of shared memory)",
)
    parser.add_argument("--subset-ratio", type=float, default=PROBE_SUBSET_RATIO, help="Probe data subset ratio")
    parser.add_argument("--elo-quantile", type=float, default=ELO_QUANTILE, help="Elo top/bottom quantile for binary probe")
    parser.add_argument("--seed", type=int, default=PROBE_RANDOM_SEED, help="Random seed for probes")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"), help="Where to save best MAE checkpoint")
    parser.add_argument(
        "--architecture",
        type=str,
        default=DEFAULT_ARCHITECTURE_ID,
        help=f"Registered architecture id (default {DEFAULT_ARCHITECTURE_ID})",
    )
    parser.add_argument(
        "--arch-config",
        type=str,
        default=None,
        help="Architecture params: path to JSON file or inline JSON object",
    )
    parser.add_argument("--register", action="store_true", help="Register best checkpoint in embedding artifacts registry")
    parser.add_argument("--model-name", type=str, default=None, help="Registry name (default: auto-generated)")
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=ARTIFACTS_DIR,
        help=f"Registry root (default {ARTIFACTS_DIR})",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--no-in-memory", action="store_true", help="Load from HDF5 on each access (slower, use if RAM is tight)")
    parser.add_argument("--compile", action="store_true", dest="use_compile", help="Use torch.compile(model) for faster forward/backward")
    parser.add_argument("--profile", action="store_true", help="Run profiler on first epoch and print top ops")
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

    arch_cfg_user = parse_arch_config_arg(args.arch_config)
    print("Training MAE...", file=sys.stderr)
    model, train_losses, val_losses, train_info = run_mae_training(
        args.train_h5,
        args.val_h5,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        workers=args.workers,
        use_amp=use_amp,
        checkpoint_dir=args.checkpoint_dir,
        in_memory=not args.no_in_memory,
        use_compile=args.use_compile,
        profile=args.profile,
        architecture_id=args.architecture,
        architecture_config=arch_cfg_user,
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

    def get_len(p: Path) -> int:
        with h5py.File(p, "r") as f:
            return f["board"].shape[0]

    n_train = get_len(args.train_h5)
    n_val = get_len(args.val_h5)
    n_test = get_len(args.test_h5)
    train_idx = _subsample_indices(n_train, args.subset_ratio, args.seed)
    val_idx = _subsample_indices(n_val, args.subset_ratio, args.seed + 1)
    test_idx = _subsample_indices(n_test, args.subset_ratio, args.seed + 2)

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

    print("Running probes (random embedding baseline)...", file=sys.stderr)
    rng = np.random.default_rng(args.seed)
    emb_dim = int(getattr(model, "embedding_dim", train_info["architecture_config"].get("embedding_dim", 128)))
    X_train_re = rng.standard_normal((len(train_idx), emb_dim)).astype(np.float32)
    rng_val = np.random.default_rng(args.seed + 1)
    X_val_re = rng_val.standard_normal((len(val_idx), emb_dim)).astype(np.float32)
    rng_test = np.random.default_rng(args.seed + 2)
    X_test_re = rng_test.standard_normal((len(test_idx), emb_dim)).astype(np.float32)
    _, meta_train = extract_embeddings(model, args.train_h5, train_idx, device)
    _, meta_val = extract_embeddings(model, args.val_h5, val_idx, device)
    _, meta_test = extract_embeddings(model, args.test_h5, test_idx, device)
    probe_results_random_emb = run_probes_on_embeddings(
        X_train_re, X_val_re, X_test_re, meta_train, meta_val, meta_test,
        device, seed=args.seed, elo_quantile=args.elo_quantile,
    )

    print("Running probes (random-weights model baseline)...", file=sys.stderr)
    model_random = build_model(train_info["architecture_id"], train_info["architecture_config"]).to(device)
    probe_results_random_model = run_probes(
        model_random,
        args.train_h5,
        args.val_h5,
        args.test_h5,
        device,
        subset_ratio=args.subset_ratio,
        elo_quantile=args.elo_quantile,
        seed=args.seed,
    )

    print("Running probes (raw input baseline)...", file=sys.stderr)
    X_train_raw, meta_train = extract_raw_inputs(args.train_h5, train_idx)
    X_val_raw, meta_val = extract_raw_inputs(args.val_h5, val_idx)
    X_test_raw, meta_test = extract_raw_inputs(args.test_h5, test_idx)
    # Raw input is (n, 1216); keep probe training on CPU to avoid GPU OOM
    probe_results_raw_input = run_probes_on_embeddings(
        X_train_raw, X_val_raw, X_test_raw, meta_train, meta_val, meta_test,
        torch.device("cpu"),
        seed=args.seed,
        elo_quantile=args.elo_quantile,
    )

    if args.register:
        repo_root = Path(__file__).resolve().parents[1]
        ckpt_path = args.checkpoint_dir / "best.pt"
        if not ckpt_path.is_file():
            print(f"Error: --register set but no checkpoint at {ckpt_path}", file=sys.stderr)
            return 1
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        name = args.model_name or generate_model_name(repo_root=repo_root, artifacts_dir=args.artifacts_dir)
        register_model(
            name=name,
            architecture_id=state["architecture_id"],
            architecture_config=state["architecture_config"],
            train_meta=state["train_meta"],
            train_hparams=state["train_hparams"],
            checkpoint_payload=state,
            repo_root=repo_root,
            artifacts_dir=args.artifacts_dir,
        )
        print(f"Registered model as {name!r}", file=sys.stderr)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_report(
        args.report,
        mae_train_losses=train_losses,
        mae_val_losses=val_losses,
        mae_test_loss=mae_test_loss,
        mae_baseline_loss=mae_baseline_loss,
        mae_improvement_pct=mae_improvement_pct,
        probe_results_final=probe_results_final,
        probe_results_random_emb=probe_results_random_emb,
        probe_results_random_model=probe_results_random_model,
        probe_results_raw_input=probe_results_raw_input,
    )
    print(f"Report written to {args.report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
