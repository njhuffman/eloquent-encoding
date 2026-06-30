"""Dataset over jepa3-packed move rows (reused on-disk format). Lazy h5 read, optional fixed subsample."""
from __future__ import annotations
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

_FIELDS_U8 = ("from_sq", "to_sq", "promotion")

# Absent-ply sentinel value (int8 -1 = 255 on disk; we cast to int64 -1 in Python).
_HIST_ABSENT_SQ = -1
_HIST_ABSENT_CAP = 0
_HIST_LEN = 4


class PackedMoveDataset(Dataset):
    def __init__(self, h5_path: str | Path, *, sample_n: int | None = None, seed: int = 0,
                 band: tuple[int, int] | None = None, sequential: bool = False):
        self.path = str(h5_path)
        with h5py.File(self.path, "r") as f:
            n = int(f["packed_pre"].shape[0])
            if band is not None:
                elo = f["elo_to_move"][:]
                pool = np.nonzero((elo >= band[0]) & (elo < band[1]))[0]
            else:
                pool = np.arange(n)
            # Detect history columns once at construction time (not per-row).
            self._has_hist: bool = "hist_from" in f
        if sample_n is not None and sample_n < len(pool):
            if sequential:
                # Pre-shuffled-on-disk file: take the first N in order (zero random reads).
                self.indices = pool[:sample_n]
            else:
                rng = np.random.default_rng(seed)
                self.indices = np.sort(rng.choice(pool, size=sample_n, replace=False))
        else:
            self.indices = pool  # nonzero()/arange() are already ascending
        self._f: h5py.File | None = None

    def _file(self) -> h5py.File:
        if self._f is None:
            self._f = h5py.File(self.path, "r")  # opened per-worker
        return self._f

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[i])
        f = self._file()
        out = {
            "packed_pre": torch.from_numpy(f["packed_pre"][idx].astype(np.uint8)),
            "from_legal_u64": torch.from_numpy(np.array(f["from_legal_u64"][idx], dtype=np.uint64)).to(torch.int64),
            "to_legal_u64": torch.from_numpy(np.array(f["to_legal_u64"][idx], dtype=np.uint64)).to(torch.int64),
            "elo_to_move": torch.tensor(int(f["elo_to_move"][idx]), dtype=torch.int64),
            "result": torch.tensor(int(f["result"][idx]), dtype=torch.int64),
            "opp_elo": torch.tensor(int(f["opp_elo"][idx]), dtype=torch.int64),
        }
        for k in _FIELDS_U8:
            out[k] = torch.tensor(int(f[k][idx]), dtype=torch.int64)
        # Optional last-move history columns (absent-by-default for older datasets).
        if self._has_hist:
            out["hist_from"] = torch.from_numpy(f["hist_from"][idx].astype(np.int64))
            out["hist_to"]   = torch.from_numpy(f["hist_to"][idx].astype(np.int64))
            out["hist_cap"]  = torch.from_numpy(f["hist_cap"][idx].astype(np.int64))
        else:
            out["hist_from"] = torch.full((_HIST_LEN,), _HIST_ABSENT_SQ,  dtype=torch.int64)
            out["hist_to"]   = torch.full((_HIST_LEN,), _HIST_ABSENT_SQ,  dtype=torch.int64)
            out["hist_cap"]  = torch.full((_HIST_LEN,), _HIST_ABSENT_CAP, dtype=torch.int64)
        return out

    @staticmethod
    def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        keys = batch[0].keys()
        return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
