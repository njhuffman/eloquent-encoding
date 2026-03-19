#!/usr/bin/env python3
"""
Linear probes to validate the chess embedding: piece count, in_check, elo (regression + top/bottom N%).
Uses single-layer linear MLPs (trained with MSE or BCE) on a subset of train/val/test.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from embedding.config import (
    BOARD_CHANNELS,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    ELO_QUANTILE,
    PROBE_EPOCHS,
    PROBE_LR,
    PROBE_RANDOM_SEED,
    PROBE_SUBSET_RATIO,
)
from embedding.model import ChessMAE
from embedding.probes import (
    predict_classifier_proba,
    predict_regression,
    train_classifier_probe,
    train_regression_probe,
)


def _subsample_indices(n_total: int, ratio: float, seed: int) -> np.ndarray:
    """Return indices for a random subset of size ratio * n_total."""
    n = max(1, int(round(ratio * n_total)))
    rng = np.random.default_rng(seed)
    return rng.choice(n_total, size=min(n, n_total), replace=False)


def _encoder_input_from_board(board: np.ndarray) -> np.ndarray:
    """Full board (8,8,18) -> encoder input (8,8,19) with zero mask (no masking)."""
    mask = np.zeros((*board.shape[:2], 1), dtype=np.float32)
    return np.concatenate([board, mask], axis=-1)


def extract_embeddings(
    model: ChessMAE,
    h5_path: Path,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load boards and meta at indices from HDF5, run encoder (full board, no masking), return (embeddings, meta).
    embeddings: (len(indices), 128), meta: (len(indices), 6).
    """
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
            # (B, 8, 8, 19)
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run linear probes to validate embeddings")
    parser.add_argument("--train-h5", type=Path, required=True, help="Train HDF5")
    parser.add_argument("--val-h5", type=Path, required=True, help="Val HDF5")
    parser.add_argument("--test-h5", type=Path, required=True, help="Test HDF5")
    parser.add_argument("--checkpoint", type=Path, default=None, help="MAE checkpoint (default: checkpoints/best.pt)")
    parser.add_argument("--subset-ratio", type=float, default=PROBE_SUBSET_RATIO, help=f"Fraction of data to use (default {PROBE_SUBSET_RATIO})")
    parser.add_argument("--elo-quantile", type=float, default=ELO_QUANTILE, help=f"Top/bottom N%% for elo binary probe (default {ELO_QUANTILE})")
    parser.add_argument("--seed", type=int, default=PROBE_RANDOM_SEED, help="Random seed for subsampling")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    for p in (args.train_h5, args.val_h5, args.test_h5):
        if not p.exists():
            print(f"Error: not found: {p}", file=sys.stderr)
            return 1

    ckpt = args.checkpoint or (_REPO_ROOT / "checkpoints" / "best.pt")
    if not ckpt.exists():
        print(f"Error: checkpoint not found: {ckpt}", file=sys.stderr)
        return 1

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model = ChessMAE()
    model.load_state_dict(state["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    def get_len(h5_path: Path) -> int:
        with h5py.File(h5_path, "r") as f:
            return f["board"].shape[0]

    n_train = get_len(args.train_h5)
    n_val = get_len(args.val_h5)
    n_test = get_len(args.test_h5)
    train_idx = _subsample_indices(n_train, args.subset_ratio, args.seed)
    val_idx = _subsample_indices(n_val, args.subset_ratio, args.seed + 1)
    test_idx = _subsample_indices(n_test, args.subset_ratio, args.seed + 2)

    print("Extracting embeddings (subset)...", file=sys.stderr)
    X_train, meta_train = extract_embeddings(model, args.train_h5, train_idx, device)
    X_val, meta_val = extract_embeddings(model, args.val_h5, val_idx, device)
    X_test, meta_test = extract_embeddings(model, args.test_h5, test_idx, device)
    # meta columns: elo_white, elo_black, piece_count_white, piece_count_black, outcome, in_check
    print(f"Train {X_train.shape[0]}, Val {X_val.shape[0]}, Test {X_test.shape[0]}", file=sys.stderr)

    results = []

    # --- 1. Piece count (regression): white and black
    for probe_name, col in [("piece_count_white", 2), ("piece_count_black", 3)]:
        y_train = meta_train[:, col]
        y_val = meta_val[:, col]
        y_test = meta_test[:, col]
        probe = train_regression_probe(
            X_train, y_train, device, seed=args.seed, epochs=PROBE_EPOCHS, lr=PROBE_LR
        )
        for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
            pred = predict_regression(probe, X, device)
            r2 = r2_score(y, pred)
            mse = mean_squared_error(y, pred)
            results.append((probe_name, split_name, {"r2": r2, "mse": mse}))
            print(f"Probe {probe_name} [{split_name}] R2={r2:.4f} MSE={mse:.4f}", file=sys.stderr)

    # --- 2. in_check (binary)
    y_train = (meta_train[:, 5] >= 0.5).astype(np.int64)
    y_val = (meta_val[:, 5] >= 0.5).astype(np.int64)
    y_test = (meta_test[:, 5] >= 0.5).astype(np.int64)
    probe = train_classifier_probe(
        X_train, y_train, device, seed=args.seed, epochs=PROBE_EPOCHS, lr=PROBE_LR
    )
    for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        proba = predict_classifier_proba(probe, X, device)
        pred = (proba >= 0.5).astype(int)
        acc = accuracy_score(y, pred)
        try:
            auc = roc_auc_score(y, proba)
        except ValueError:
            auc = float("nan")
        results.append(("in_check", split_name, {"accuracy": acc, "roc_auc": auc}))
        print(f"Probe in_check [{split_name}] accuracy={acc:.4f} roc_auc={auc:.4f}", file=sys.stderr)

    # --- 3. Elo regression (mean of white and black)
    mean_elo_train = (meta_train[:, 0] + meta_train[:, 1]) / 2
    mean_elo_val = (meta_val[:, 0] + meta_val[:, 1]) / 2
    mean_elo_test = (meta_test[:, 0] + meta_test[:, 1]) / 2
    probe = train_regression_probe(
        X_train, mean_elo_train, device, seed=args.seed, epochs=PROBE_EPOCHS, lr=PROBE_LR
    )
    for split_name, X, y in [("val", X_val, mean_elo_val), ("test", X_test, mean_elo_test)]:
        pred = predict_regression(probe, X, device)
        r2 = r2_score(y, pred)
        mse = mean_squared_error(y, pred)
        results.append(("elo_regression", split_name, {"r2": r2, "mse": mse}))
        print(f"Probe elo_regression [{split_name}] R2={r2:.4f} MSE={mse:.4f}", file=sys.stderr)

    # --- 4. Elo top N% vs bottom N%
    q_lo = args.elo_quantile
    q_hi = 1.0 - args.elo_quantile
    thresh_lo = np.quantile(mean_elo_train, q_lo)
    thresh_hi = np.quantile(mean_elo_train, q_hi)
    train_mask = (mean_elo_train <= thresh_lo) | (mean_elo_train >= thresh_hi)
    y_train_bin = (mean_elo_train >= thresh_hi).astype(np.int64)
    X_train_bin = X_train[train_mask]
    y_train_bin = y_train_bin[train_mask]
    y_val_bin = (mean_elo_val >= thresh_hi).astype(np.int64)
    y_test_bin = (mean_elo_test >= thresh_hi).astype(np.int64)
    probe = train_classifier_probe(
        X_train_bin, y_train_bin, device, seed=args.seed, epochs=PROBE_EPOCHS, lr=PROBE_LR
    )
    for split_name, X, y in [("val", X_val, y_val_bin), ("test", X_test, y_test_bin)]:
        proba = predict_classifier_proba(probe, X, device)
        pred = (proba >= 0.5).astype(int)
        acc = accuracy_score(y, pred)
        try:
            auc = roc_auc_score(y, proba)
        except ValueError:
            auc = float("nan")
        results.append(("elo_top_vs_bottom", split_name, {"accuracy": acc, "roc_auc": auc}))
        print(f"Probe elo_top_vs_bottom [{split_name}] accuracy={acc:.4f} roc_auc={auc:.4f}", file=sys.stderr)

    print("\n--- Summary ---", file=sys.stderr)
    for probe_name, split_name, metrics in results:
        print(f"  {probe_name} [{split_name}]: {metrics}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
