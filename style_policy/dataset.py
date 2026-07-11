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
_NNUE_NA = -32768                      # Stockfish-eval NA sentinel; must match STATIC_NA in stockfish_eval.py


class PackedMoveDataset(Dataset):
    def __init__(self, h5_path: str | Path, *, sample_n: int | None = None, seed: int = 0,
                 band: tuple[int, int] | None = None, sequential: bool = False, preload: bool = False,
                 nnue_path: str | Path | None = None):
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
            # Detect Maia-3 soft-target columns (distillation label sets only).
            self._has_soft: bool = "maia_from" in f
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
        # Optional RAM preload: load full fields into memory (index by h5-row) -> no per-row h5
        # reads. Use num_workers=0 (else each worker duplicates the arrays -> OOM).
        self._ram: dict | None = None
        if preload:
            _keys = ["packed_pre", "from_legal_u64", "to_legal_u64", "elo_to_move", "result",
                     "opp_elo", "from_sq", "to_sq", "promotion"]
            if self._has_hist: _keys += ["hist_from", "hist_to", "hist_cap"]
            if self._has_soft: _keys += ["maia_from", "maia_to"]
            with h5py.File(self.path, "r") as f:
                self._ram = {k: f[k][:] for k in _keys}
        # Optional Stockfish static-NNUE eval sidecar (row-aligned) for the multi-task NNUE head.
        self._nnue_path = str(nnue_path) if nnue_path is not None else None
        self._nnue_ram: np.ndarray | None = None
        self._nnue_f: h5py.File | None = None
        if self._nnue_path is not None and preload:
            with h5py.File(self._nnue_path, "r") as nf:
                self._nnue_ram = nf["sf_cp"][:]

    def _file(self) -> h5py.File:
        if self._f is None:
            self._f = h5py.File(self.path, "r")  # opened per-worker
        return self._f

    def _nnue_file(self) -> h5py.File:
        if self._nnue_f is None:
            self._nnue_f = h5py.File(self._nnue_path, "r")
        return self._nnue_f

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[i])
        src = self._ram if self._ram is not None else self._file()
        out = {
            "packed_pre": torch.from_numpy(src["packed_pre"][idx].astype(np.uint8)),
            "from_legal_u64": torch.from_numpy(np.array(src["from_legal_u64"][idx], dtype=np.uint64)).to(torch.int64),
            "to_legal_u64": torch.from_numpy(np.array(src["to_legal_u64"][idx], dtype=np.uint64)).to(torch.int64),
            "elo_to_move": torch.tensor(int(src["elo_to_move"][idx]), dtype=torch.int64),
            "result": torch.tensor(int(src["result"][idx]), dtype=torch.int64),
            "opp_elo": torch.tensor(int(src["opp_elo"][idx]), dtype=torch.int64),
        }
        for k in _FIELDS_U8:
            out[k] = torch.tensor(int(src[k][idx]), dtype=torch.int64)
        # Optional last-move history columns (absent-by-default for older datasets).
        if self._has_hist:
            out["hist_from"] = torch.from_numpy(src["hist_from"][idx].astype(np.int64))
            out["hist_to"]   = torch.from_numpy(src["hist_to"][idx].astype(np.int64))
            out["hist_cap"]  = torch.from_numpy(src["hist_cap"][idx].astype(np.int64))
        else:
            out["hist_from"] = torch.full((_HIST_LEN,), _HIST_ABSENT_SQ,  dtype=torch.int64)
            out["hist_to"]   = torch.full((_HIST_LEN,), _HIST_ABSENT_SQ,  dtype=torch.int64)
            out["hist_cap"]  = torch.full((_HIST_LEN,), _HIST_ABSENT_CAP, dtype=torch.int64)
        # Optional Maia-3 soft targets (P(from) and P(to|true-from), 64-vectors) for distillation.
        if self._has_soft:
            out["maia_from"] = torch.from_numpy(src["maia_from"][idx].astype(np.float32))
            out["maia_to"]   = torch.from_numpy(src["maia_to"][idx].astype(np.float32))
        if self._nnue_path is not None:
            cp = int(self._nnue_ram[idx]) if self._nnue_ram is not None else int(self._nnue_file()["sf_cp"][idx])
            valid = cp != _NNUE_NA
            out["nnue_value"] = torch.tensor(float(np.tanh(cp / 400.0)) if valid else 0.0, dtype=torch.float32)
            out["nnue_valid"] = torch.tensor(valid, dtype=torch.bool)
        return out

    @staticmethod
    def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        keys = batch[0].keys()
        return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
