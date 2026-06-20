#!/usr/bin/env python3
"""
Linear probes on the JEPA **online encoder** latent (board_t -> encoder_online).

Targets mirror embedding/scripts/run_probes.py where possible:
  - piece_count_white / piece_count_black (from board_t tensor)
  - in_check (from board_t via python-chess)
  - elo_regression / elo_top_vs_bottom on the HDF5 **mover Elo** column
    (embedding uses mean(white_elo, black_elo) from meta; JEPA stores only mover Elo.)

Requires JEPA-format HDF5: board_t, elo (and standard neg datasets; unused here).
"""

from __future__ import annotations

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

from jepa.architectures import DEFAULT_ARCHITECTURE_ID, build_model
from jepa.board_labels import in_check_batch, piece_counts_batch
from jepa.config import (
    ELO_QUANTILE,
    PROBE_EPOCHS,
    PROBE_LR,
    PROBE_RANDOM_SEED,
    PROBE_SUBSET_RATIO,
)
from jepa.load import load_jepa_by_name, load_jepa_from_checkpoint
from jepa.probes import (
    predict_classifier_proba,
    predict_regression,
    train_classifier_probe,
    train_regression_probe,
)


def _subsample_indices(n_total: int, ratio: float, seed: int) -> np.ndarray:
    n = max(1, int(round(ratio * n_total)))
    rng = np.random.default_rng(seed)
    return rng.choice(n_total, size=min(n, n_total), replace=False)


def extract_encoder_embeddings(
    model: torch.nn.Module,
    h5_path: Path,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (embeddings, elo, boards) for rows at indices.
    embeddings: (len(indices), D), elo: (len(indices),), boards: (len(indices), 8, 8, 18).
    """
    model.eval()
    indices = np.asarray(indices)
    emb_list: list[np.ndarray] = []
    elo_list: list[np.ndarray] = []
    board_list: list[np.ndarray] = []
    enc = model.encoder_online
    with h5py.File(h5_path, "r") as f:
        if "board_t" not in f or "elo" not in f:
            raise ValueError(f"{h5_path} must contain 'board_t' and 'elo' (JEPA schema).")
        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            batch_idx = indices[start:end]
            boards = np.stack([f["board_t"][i] for i in batch_idx], axis=0).astype(np.float32)
            elo_b = np.stack([float(f["elo"][i]) for i in batch_idx], axis=0).astype(np.float32)
            with torch.no_grad():
                x = torch.from_numpy(boards).to(device)
                z = enc(x)
                z = z.cpu().numpy()
            emb_list.append(z)
            elo_list.append(elo_b)
            board_list.append(boards)
    return np.vstack(emb_list), np.concatenate(elo_list), np.vstack(board_list)


def main() -> int:
    parser = argparse.ArgumentParser(description="Linear probes for JEPA encoder latents")
    parser.add_argument("--train-h5", type=Path, required=True)
    parser.add_argument("--val-h5", type=Path, required=True)
    parser.add_argument("--test-h5", type=Path, required=True)
    parser.add_argument(
        "--jepa-model",
        type=str,
        default=None,
        help="Registered model name (jepa/artifacts registry); overrides --checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="JEPA checkpoint .pt (default: jepa_checkpoints/best.pt under repo)",
    )
    parser.add_argument("--subset-ratio", type=float, default=PROBE_SUBSET_RATIO)
    parser.add_argument("--elo-quantile", type=float, default=ELO_QUANTILE)
    parser.add_argument("--seed", type=int, default=PROBE_RANDOM_SEED)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    for p in (args.train_h5, args.val_h5, args.test_h5):
        if not p.exists():
            print(f"Error: not found: {p}", file=sys.stderr)
            return 1

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.jepa_model:
        try:
            model = load_jepa_by_name(args.jepa_model, repo_root=_REPO_ROOT, device=device)
        except Exception as e:
            print(f"Error loading registered JEPA model {args.jepa_model!r}: {e}", file=sys.stderr)
            return 1
    else:
        ckpt = args.checkpoint or (_REPO_ROOT / "jepa_checkpoints" / "best.pt")
        if not ckpt.is_file():
            print(f"Error: checkpoint not found: {ckpt}", file=sys.stderr)
            return 1
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if state.get("architecture_id"):
            model = load_jepa_from_checkpoint(ckpt, device=device)
        else:
            model = build_model(DEFAULT_ARCHITECTURE_ID, {})
            model.load_state_dict(state["model_state_dict"], strict=True)
            model = model.to(device)
    model.eval()

    def get_len(h5_path: Path) -> int:
        with h5py.File(h5_path, "r") as f:
            return f["board_t"].shape[0]

    n_train = get_len(args.train_h5)
    n_val = get_len(args.val_h5)
    n_test = get_len(args.test_h5)
    train_idx = _subsample_indices(n_train, args.subset_ratio, args.seed)
    val_idx = _subsample_indices(n_val, args.subset_ratio, args.seed + 1)
    test_idx = _subsample_indices(n_test, args.subset_ratio, args.seed + 2)

    print("Extracting JEPA encoder embeddings (subset)...", file=sys.stderr)
    X_train, elo_train, boards_train = extract_encoder_embeddings(
        model, args.train_h5, train_idx, device, batch_size=args.batch_size
    )
    X_val, elo_val, boards_val = extract_encoder_embeddings(
        model, args.val_h5, val_idx, device, batch_size=args.batch_size
    )
    X_test, elo_test, boards_test = extract_encoder_embeddings(
        model, args.test_h5, test_idx, device, batch_size=args.batch_size
    )

    pcw_tr, pcb_tr = piece_counts_batch(boards_train)
    pcw_va, pcb_va = piece_counts_batch(boards_val)
    pcw_te, pcb_te = piece_counts_batch(boards_test)

    print(f"Train {X_train.shape[0]}, Val {X_val.shape[0]}, Test {X_test.shape[0]}", file=sys.stderr)
    print(f"Embedding dim {X_train.shape[1]}", file=sys.stderr)

    results: list[tuple[str, str, dict]] = []

    for probe_name, y_train, y_val, y_test in [
        ("piece_count_white", pcw_tr, pcw_va, pcw_te),
        ("piece_count_black", pcb_tr, pcb_va, pcb_te),
    ]:
        probe = train_regression_probe(
            X_train, y_train, device, seed=args.seed, epochs=PROBE_EPOCHS, lr=PROBE_LR
        )
        for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
            pred = predict_regression(probe, X, device)
            r2 = r2_score(y, pred)
            mse = mean_squared_error(y, pred)
            results.append((probe_name, split_name, {"r2": r2, "mse": mse}))
            print(f"Probe {probe_name} [{split_name}] R2={r2:.4f} MSE={mse:.4f}", file=sys.stderr)

    print("Computing in_check labels (board_t)...", file=sys.stderr)
    y_train_ic = in_check_batch(boards_train)
    y_val_ic = in_check_batch(boards_val)
    y_test_ic = in_check_batch(boards_test)
    y_train_ic_i = (y_train_ic >= 0.5).astype(np.int64)
    y_val_ic_i = (y_val_ic >= 0.5).astype(np.int64)
    y_test_ic_i = (y_test_ic >= 0.5).astype(np.int64)

    probe = train_classifier_probe(
        X_train, y_train_ic_i, device, seed=args.seed, epochs=PROBE_EPOCHS, lr=PROBE_LR
    )
    for split_name, X, y in [("val", X_val, y_val_ic_i), ("test", X_test, y_test_ic_i)]:
        proba = predict_classifier_proba(probe, X, device)
        pred = (proba >= 0.5).astype(int)
        acc = accuracy_score(y, pred)
        try:
            auc = roc_auc_score(y, proba)
        except ValueError:
            auc = float("nan")
        results.append(("in_check", split_name, {"accuracy": acc, "roc_auc": auc}))
        print(f"Probe in_check [{split_name}] accuracy={acc:.4f} roc_auc={auc:.4f}", file=sys.stderr)

    probe = train_regression_probe(
        X_train, elo_train, device, seed=args.seed, epochs=PROBE_EPOCHS, lr=PROBE_LR
    )
    for split_name, X, y in [("val", X_val, elo_val), ("test", X_test, elo_test)]:
        pred = predict_regression(probe, X, device)
        r2 = r2_score(y, pred)
        mse = mean_squared_error(y, pred)
        results.append(("elo_regression", split_name, {"r2": r2, "mse": mse}))
        print(f"Probe elo_regression (mover Elo) [{split_name}] R2={r2:.4f} MSE={mse:.4f}", file=sys.stderr)

    q_lo = args.elo_quantile
    q_hi = 1.0 - args.elo_quantile
    thresh_lo = np.quantile(elo_train, q_lo)
    thresh_hi = np.quantile(elo_train, q_hi)
    train_mask = (elo_train <= thresh_lo) | (elo_train >= thresh_hi)
    y_train_bin = (elo_train >= thresh_hi).astype(np.int64)
    X_train_bin = X_train[train_mask]
    y_train_bin = y_train_bin[train_mask]
    y_val_bin = (elo_val >= thresh_hi).astype(np.int64)
    y_test_bin = (elo_test >= thresh_hi).astype(np.int64)
    if X_train_bin.shape[0] < 2:
        print("Warning: too few train rows for elo_top_vs_bottom after quantile mask.", file=sys.stderr)
    else:
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
            print(
                f"Probe elo_top_vs_bottom (mover Elo) [{split_name}] accuracy={acc:.4f} roc_auc={auc:.4f}",
                file=sys.stderr,
            )

    print("\n--- Summary ---", file=sys.stderr)
    for probe_name, split_name, metrics in results:
        print(f"  {probe_name} [{split_name}]: {metrics}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
