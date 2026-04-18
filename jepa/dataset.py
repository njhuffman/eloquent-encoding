"""
PyTorch Dataset for Chess-JEPA: board_t, next positive, K negatives, mover Elo (RAM-backed).

Training materializes float32 arrays once per stage; ``ChessJEPADatasetInMemory`` serves batches
without HDF5 I/O for those tensors.
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from jepa.config import BOARD_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH


class ChessJEPADatasetInMemory(Dataset):
    """Preloaded arrays; no I/O in __getitem__."""

    def __init__(
        self,
        board_t: np.ndarray,
        board_pos: np.ndarray,
        board_negs: np.ndarray,
        elo: np.ndarray,
    ):
        n = board_t.shape[0]
        assert board_pos.shape[0] == n and board_negs.shape[0] == n and elo.shape[0] == n
        assert board_t.shape[1:] == (BOARD_HEIGHT, BOARD_WIDTH, BOARD_CHANNELS)
        self._board_t = board_t
        self._board_pos = board_pos
        self._board_negs = board_negs
        self._elo = elo

    def __len__(self) -> int:
        return int(self._board_t.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(np.asarray(self._board_t[idx], dtype=np.float32)),
            torch.from_numpy(np.asarray(self._board_pos[idx], dtype=np.float32)),
            torch.from_numpy(np.asarray(self._board_negs[idx], dtype=np.float32)),
            torch.tensor(float(self._elo[idx]), dtype=torch.float32),
        )


def _multiprocessing_context(num_workers: int):
    """
    On Linux use ``forkserver`` so workers fork from a clean helper after the parent has used
    other native libs. Elsewhere use ``spawn``.
    """
    if num_workers <= 0:
        return None
    import sys

    import torch.multiprocessing as mp

    if sys.platform == "linux":
        try:
            return mp.get_context("forkserver")
        except ValueError:
            pass
    return mp.get_context("spawn")


def get_dataloaders_from_arrays(
    train_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    val_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Train / val loaders from pre-materialized JEPA float32 arrays."""
    tr_bt, tr_pos, tr_neg, tr_elo = train_arrays
    va_bt, va_pos, va_neg, va_elo = val_arrays
    mem_gb = (tr_bt.nbytes + tr_pos.nbytes + tr_neg.nbytes + tr_elo.nbytes) / 1e9
    mem_gb += (va_bt.nbytes + va_pos.nbytes + va_neg.nbytes + va_elo.nbytes) / 1e9
    print(f"JEPA dataloaders: in-RAM tensors (~{mem_gb:.2f} GB), workers={num_workers}", file=sys.stderr)

    train_ds = ChessJEPADatasetInMemory(tr_bt, tr_pos, tr_neg, tr_elo)
    val_ds = ChessJEPADatasetInMemory(va_bt, va_pos, va_neg, va_elo)
    loader_kw: dict = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    ctx = _multiprocessing_context(num_workers)
    if ctx is not None:
        loader_kw["multiprocessing_context"] = ctx
    if num_workers > 0:
        loader_kw["prefetch_factor"] = 2
        loader_kw["persistent_workers"] = True
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kw)
    return train_loader, val_loader
