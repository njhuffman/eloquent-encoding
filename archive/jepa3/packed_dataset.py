"""Random-access packed move HDF5 for jepa3 (no FEN parsing at load time)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from jepa3.packed_board_codec import packed_to_board_tensor, u64_pair_to_masks
from jepa3.packed_h5 import (
    DATASET_ELO,
    DATASET_FROM_LEGAL_U64,
    DATASET_FROM_SQ,
    DATASET_PACKED_POST,
    DATASET_PACKED_PRE,
    DATASET_PROMOTION,
    DATASET_TO_LEGAL_U64,
    DATASET_TO_SQ,
    assert_packed_h5,
)


def sample_row_indices(n_total: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    if n >= n_total:
        return np.arange(n_total, dtype=np.int64)
    return np.sort(rng.choice(n_total, size=n, replace=False).astype(np.int64))


class PackedMoveRowDataset(Dataset):
    """Random-access rows by index from jepa3 packed HDF5."""

    def __init__(self, h5_path: Path | str, indices: np.ndarray) -> None:
        self.h5_path = Path(h5_path)
        assert_packed_h5(self.h5_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self._f: h5py.File | None = None

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def _file(self) -> h5py.File:
        if self._f is None:
            self._f = h5py.File(self.h5_path, "r", swmr=True)
        return self._f

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int, int, int]:
        idx = int(self.indices[i])
        f = self._file()
        p_pre = np.asarray(f[DATASET_PACKED_PRE][idx], dtype=np.uint8)
        p_post = np.asarray(f[DATASET_PACKED_POST][idx], dtype=np.uint8)
        fu = int(f[DATASET_FROM_LEGAL_U64][idx])
        tu = int(f[DATASET_TO_LEGAL_U64][idx])
        fs = int(f[DATASET_FROM_SQ][idx])
        ts = int(f[DATASET_TO_SQ][idx])
        pr = int(f[DATASET_PROMOTION][idx])
        elo = float(f[DATASET_ELO][idx])

        board_t = packed_to_board_tensor(p_pre)
        board_post = packed_to_board_tensor(p_post)
        from_m, to_m = u64_pair_to_masks(np.uint64(fu), np.uint64(tu))
        return board_t, board_post, from_m, to_m, elo, fs, ts, pr

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_f"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)


def collate_packed_move_rows(
    batch: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int, int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    boards = torch.stack([torch.from_numpy(b[0]) for b in batch], dim=0)
    post = torch.stack([torch.from_numpy(b[1]) for b in batch], dim=0)
    from_m = torch.stack([torch.from_numpy(b[2]) for b in batch], dim=0)
    to_m = torch.stack([torch.from_numpy(b[3]) for b in batch], dim=0)
    elo = torch.tensor([b[4] for b in batch], dtype=torch.float32)
    fs = torch.tensor([b[5] for b in batch], dtype=torch.long)
    ts = torch.tensor([b[6] for b in batch], dtype=torch.long)
    pr = torch.tensor([b[7] for b in batch], dtype=torch.long)
    return boards, post, from_m, to_m, elo, fs, ts, pr


def make_packed_loader(
    h5_path: Path | str,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int | None = None,
) -> DataLoader:
    ds = PackedMoveRowDataset(h5_path, indices)
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(num_workers),
        collate_fn=collate_packed_move_rows,
        drop_last=shuffle,
        generator=gen if shuffle else None,
    )
