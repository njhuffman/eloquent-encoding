"""Streaming move-row dataset from HDF5 (no pre-materialized negatives)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from jepa.move_row_codec import tensors_for_row


class MoveRowDataset(Dataset):
    """Random-access rows by index from a move-sample HDF5."""

    def __init__(self, h5_path: Path | str, indices: np.ndarray) -> None:
        self.h5_path = Path(h5_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self._f: h5py.File | None = None

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def _file(self) -> h5py.File:
        if self._f is None:
            self._f = h5py.File(self.h5_path, "r", swmr=True)
        return self._f

    def __getitem__(self, i: int) -> tuple[np.ndarray, float, int, int, int, str]:
        idx = int(self.indices[i])
        f = self._file()
        fen = f["fen"][idx]
        if isinstance(fen, bytes):
            fen = fen.decode("utf-8")
        elo = float(f["elo_to_move"][idx])
        fs = int(f["from_sq"][idx])
        ts = int(f["to_sq"][idx])
        pr = int(f["promotion"][idx])
        t = tensors_for_row(str(fen), fs, ts, pr)
        if t is None:
            raise RuntimeError(f"invalid fen/move at row {idx}: {fen!r}")
        board_t, _ = t
        return board_t.astype(np.float32, copy=False), elo, fs, ts, pr, str(fen)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_f"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)


def collate_move_rows(
    batch: list[tuple[np.ndarray, float, int, int, int, str]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    boards = torch.stack([torch.from_numpy(b[0]) for b in batch], dim=0)
    elo = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    fs = torch.tensor([b[2] for b in batch], dtype=torch.long)
    ts = torch.tensor([b[3] for b in batch], dtype=torch.long)
    pr = torch.tensor([b[4] for b in batch], dtype=torch.long)
    fens = [b[5] for b in batch]
    return boards, elo, fs, ts, pr, fens


def make_loader(
    h5_path: Path | str,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int | None = None,
) -> DataLoader:
    ds = MoveRowDataset(h5_path, indices)
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(num_workers),
        collate_fn=collate_move_rows,
        drop_last=shuffle,
        generator=gen if shuffle else None,
    )


def sample_row_indices(n_total: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    if n >= n_total:
        return np.arange(n_total, dtype=np.int64)
    return np.sort(rng.choice(n_total, size=n, replace=False).astype(np.int64))
