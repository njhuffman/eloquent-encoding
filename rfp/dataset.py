"""HDF5 dataset for rfp training."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from jepa3.packed_board_codec import u64_to_legal_mask_float

from rfp.h5_io import (
    DATASET_DELTA_Z,
    DATASET_ELO_BUCKET,
    DATASET_FROM_LEGAL_U64,
    DATASET_FROM_SQ,
    DATASET_HISTORY_MASK,
    DATASET_Z_CURR,
    assert_rfp_h5,
)


def sample_row_indices(n_total: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    if n >= n_total:
        return np.arange(n_total, dtype=np.int64)
    return np.sort(rng.choice(n_total, size=n, replace=False).astype(np.int64))


class RfpH5Dataset(Dataset):
    """Random-access rows from an rfp HDF5 file."""

    def __init__(self, h5_path: Path | str, indices: np.ndarray) -> None:
        self.h5_path = Path(h5_path)
        assert_rfp_h5(self.h5_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self._f: h5py.File | None = None

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def _file(self) -> h5py.File:
        if self._f is None:
            self._f = h5py.File(self.h5_path, "r", swmr=True)
        return self._f

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
        idx = int(self.indices[i])
        f = self._file()
        delta_z = np.asarray(f[DATASET_DELTA_Z][idx], dtype=np.float32)
        z_curr = np.asarray(f[DATASET_Z_CURR][idx], dtype=np.float32)
        hist_mask = np.asarray(f[DATASET_HISTORY_MASK][idx], dtype=np.float32)
        fu = int(f[DATASET_FROM_LEGAL_U64][idx])
        fs = int(f[DATASET_FROM_SQ][idx])
        elo_raw = int(f[DATASET_ELO_BUCKET][idx])
        from_m = u64_to_legal_mask_float(fu)
        return delta_z, z_curr, hist_mask, from_m, fs, elo_raw

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_f"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)


def collate_rfp_batch(
    batch: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dz = torch.stack([torch.from_numpy(b[0]) for b in batch], dim=0)
    zc = torch.stack([torch.from_numpy(b[1]) for b in batch], dim=0)
    hm = torch.stack([torch.from_numpy(b[2]) for b in batch], dim=0)
    from_m = torch.stack([torch.from_numpy(b[3]) for b in batch], dim=0)
    fs = torch.tensor([b[4] for b in batch], dtype=torch.long)
    elo = torch.tensor([b[5] for b in batch], dtype=torch.long)
    return dz, zc, hm, from_m, fs, elo


def make_rfp_loader(
    h5_path: Path | str,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int | None = None,
) -> DataLoader:
    ds = RfpH5Dataset(h5_path, indices)
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(num_workers),
        collate_fn=collate_rfp_batch,
        drop_last=shuffle,
        generator=gen if shuffle else None,
    )
