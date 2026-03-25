"""
PyTorch Dataset for Chess-JEPA HDF5: board_t, next positive, K negatives, mover Elo.

Disk-backed loading follows move_predictor: one ``h5py`` file per worker, opened lazily,
``__getitems__`` + contiguous HDF5 slices for a few large reads per batch instead of
per-row I/O or loading the full file into RAM.

Read-only opens (not SWMR) and retries on transient EIO — see :func:`_is_transient_h5_read_error`.
If reads still fail on network/container storage, try ``--workers 0`` or
``HDF5_USE_FILE_LOCKING=FALSE``.
"""

from __future__ import annotations

import errno
import sys
import time
from pathlib import Path

from jepa.h5_bootstrap import apply_hdf5_read_safety_env

apply_hdf5_read_safety_env()

import h5py
import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

from jepa.config import BOARD_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH

# Retries for transient HDF5 read failures under multi-worker I/O (EIO on some mounts).
_H5_READ_RETRIES = 6
_H5_READ_BACKOFF_S = 0.05


def _is_transient_h5_read_error(exc: BaseException) -> bool:
    if not isinstance(exc, OSError):
        return False
    en = getattr(exc, "errno", None)
    if en is not None:
        if en in (errno.EIO, errno.EBUSY, errno.EINTR):
            return True
        if hasattr(errno, "ESTALE") and en == errno.ESTALE:
            return True
    msg = str(exc).lower()
    return "input/output error" in msg or "file read failed" in msg


def _read_num_negatives_k(h5_path: str | Path) -> int:
    with h5py.File(h5_path, "r") as f:
        if "board_t_plus_1_negs" not in f:
            raise ValueError(f"Missing board_t_plus_1_negs in {h5_path}")
        k = f.attrs.get("num_negatives_k")
        if k is not None:
            return int(k)
        return int(f["board_t_plus_1_negs"].shape[1])


def assert_h5_k_matches(h5_path: str | Path, expected_k: int) -> None:
    k = _read_num_negatives_k(h5_path)
    if k != expected_k:
        raise ValueError(
            f"{h5_path}: HDF5 has num_negatives_k={k} but architecture expects {expected_k}. "
            "Regenerate data or change num_negatives_k in the training spec."
        )


def load_jepa_into_memory(h5_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load full JEPA arrays into RAM. Returns board_t, pos, negs, elo."""
    with h5py.File(h5_path, "r") as f:
        bt = np.asarray(f["board_t"], dtype=np.float32)
        pos = np.asarray(f["board_t_plus_1_pos"], dtype=np.float32)
        negs = np.asarray(f["board_t_plus_1_negs"], dtype=np.float32)
        elo = np.asarray(f["elo"], dtype=np.float32)
    return bt, pos, negs, elo


class ChessJEPAH5Dataset(Dataset):
    """
    One ``h5py`` handle per worker, opened lazily (read-only, not SWMR).

    Training uses :class:`torch.utils.data.BatchSampler`; the loader calls
    ``__getitems__(indices)`` with the full index list per batch. Indices are
    sorted and merged into contiguous HDF5 row slices.
    """

    def __init__(self, file_path: str | Path):
        self.file_path = str(file_path)
        self._archive: h5py.File | None = None
        with h5py.File(self.file_path, "r") as f:
            self.length = int(f["board_t"].shape[0])

    def __len__(self) -> int:
        return self.length

    def _close_archive(self) -> None:
        if self._archive is not None:
            try:
                self._archive.close()
            except Exception:
                pass
            self._archive = None

    def _ensure_open(self) -> h5py.File:
        if self._archive is None:
            self._archive = h5py.File(self.file_path, "r")
        return self._archive

    def _read_row_slice(self, start_row: int, end_row: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sl = slice(start_row, end_row)
        last_err: BaseException | None = None
        for attempt in range(_H5_READ_RETRIES):
            try:
                f = self._ensure_open()
                bt_b = np.asarray(f["board_t"][sl], dtype=np.float32)
                pos_b = np.asarray(f["board_t_plus_1_pos"][sl], dtype=np.float32)
                negs_b = np.asarray(f["board_t_plus_1_negs"][sl], dtype=np.float32)
                elo_b = np.asarray(f["elo"][sl], dtype=np.float32)
                return bt_b, pos_b, negs_b, elo_b
            except OSError as e:
                last_err = e
                if not _is_transient_h5_read_error(e) or attempt == _H5_READ_RETRIES - 1:
                    raise
                self._close_archive()
                time.sleep(_H5_READ_BACKOFF_S * (2**attempt))
        assert last_err is not None
        raise last_err

    def __getitem__(self, index: int):
        return self.__getitems__([index])[0]

    def __getitems__(self, indices: list[int]) -> list:
        if not indices:
            return []
        idx = np.asarray(indices, dtype=np.int64)
        order = np.argsort(idx, kind="mergesort")
        sorted_idx = idx[order]
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        b = len(idx)

        sorted_bt: np.ndarray | None = None
        sorted_pos: np.ndarray | None = None
        sorted_negs: np.ndarray | None = None
        sorted_elo: np.ndarray | None = None

        pos = 0
        while pos < b:
            start_row = int(sorted_idx[pos])
            q = pos + 1
            while q < b and int(sorted_idx[q]) == int(sorted_idx[q - 1]) + 1:
                q += 1
            end_row = int(sorted_idx[q - 1]) + 1

            bt_b, pos_b, negs_b, elo_b = self._read_row_slice(start_row, end_row)

            if sorted_bt is None:
                sorted_bt = np.empty((b,) + bt_b.shape[1:], dtype=np.float32)
                sorted_pos = np.empty((b,) + pos_b.shape[1:], dtype=np.float32)
                sorted_negs = np.empty((b,) + negs_b.shape[1:], dtype=np.float32)
                sorted_elo = np.empty((b,), dtype=np.float32)

            sorted_bt[pos:q] = bt_b
            sorted_pos[pos:q] = pos_b
            sorted_negs[pos:q] = negs_b
            sorted_elo[pos:q] = elo_b
            pos = q

        assert sorted_bt is not None and sorted_pos is not None
        assert sorted_negs is not None and sorted_elo is not None

        bt_r = sorted_bt[inv]
        pos_r = sorted_pos[inv]
        negs_r = sorted_negs[inv]
        elo_r = sorted_elo[inv]

        out: list = []
        for i in range(b):
            out.append(
                (
                    torch.from_numpy(np.ascontiguousarray(bt_r[i])),
                    torch.from_numpy(np.ascontiguousarray(pos_r[i])),
                    torch.from_numpy(np.ascontiguousarray(negs_r[i])),
                    torch.tensor(float(elo_r[i]), dtype=torch.float32),
                )
            )
        return out


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
    Avoid default ``fork`` after the main process has used HDF5 (e.g. shape checks): inherited
    libhdf5 state often yields EIO under parallel reads on Linux.

    On Linux we use ``forkserver`` so workers fork from a clean helper, not from the training
    process. Elsewhere we use ``spawn`` (safe, slower startup).
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


def _make_disk_loader(
    h5_path: str | Path,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool,
    pin_memory: bool,
    multiprocessing_context=None,
) -> DataLoader:
    ds = ChessJEPAH5Dataset(h5_path)
    base_sampler = RandomSampler(ds) if shuffle else SequentialSampler(ds)
    batch_sampler = BatchSampler(base_sampler, batch_size, drop_last=drop_last)
    kw: dict = dict(
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if multiprocessing_context is not None:
        kw["multiprocessing_context"] = multiprocessing_context
    if num_workers > 0:
        kw["prefetch_factor"] = 2
        kw["persistent_workers"] = True
    return DataLoader(ds, **kw)


def get_dataloaders(
    train_h5: str | Path,
    val_h5: str | Path,
    batch_size: int,
    num_workers: int = 0,
    in_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """
    Train / val loaders. Default ``in_memory=False``: HDF5 stays on disk with batched reads
    (see :class:`ChessJEPAH5Dataset`). Set ``in_memory=True`` to preload both splits into RAM.
    """
    if in_memory:
        print("Loading train JEPA arrays into RAM...", file=sys.stderr, end=" ", flush=True)
        tr_bt, tr_pos, tr_neg, tr_elo = load_jepa_into_memory(train_h5)
        print(f"{(tr_bt.nbytes + tr_pos.nbytes + tr_neg.nbytes + tr_elo.nbytes) / 1e9:.2f} GB", file=sys.stderr)
        print("Loading val JEPA arrays into RAM...", file=sys.stderr, end=" ", flush=True)
        va_bt, va_pos, va_neg, va_elo = load_jepa_into_memory(val_h5)
        print(f"{(va_bt.nbytes + va_pos.nbytes + va_neg.nbytes + va_elo.nbytes) / 1e9:.2f} GB", file=sys.stderr)
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

    mp_ctx = _multiprocessing_context(num_workers)
    mp_note = f", mp_start={mp_ctx.get_start_method()}" if mp_ctx is not None else ""
    print(
        f"JEPA dataloaders: on-disk HDF5 (batched reads), workers={num_workers}{mp_note}",
        file=sys.stderr,
    )
    train_loader = _make_disk_loader(
        train_h5,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        multiprocessing_context=mp_ctx,
    )
    val_loader = _make_disk_loader(
        val_h5,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        multiprocessing_context=mp_ctx,
    )
    return train_loader, val_loader


def h5_transition_counts(train_h5: Path, val_h5: Path) -> tuple[int, int]:
    with h5py.File(train_h5, "r") as f:
        n_train = f["board_t"].shape[0]
    with h5py.File(val_h5, "r") as f:
        n_val = f["board_t"].shape[0]
    return n_train, n_val
