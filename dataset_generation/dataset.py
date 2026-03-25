from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class MoveSampleH5Dataset(Dataset):
    """Random-access rows written by `dataset_generation` SampleBatchWriter."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        self._archive: h5py.File | None = None
        with h5py.File(self.file_path, "r") as f:
            self._length = int(f["fen"].shape[0])

    def __len__(self) -> int:
        return self._length

    def _file(self) -> h5py.File:
        if self._archive is None:
            self._archive = h5py.File(self.file_path, "r")
        return self._archive

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        f = self._file()
        fen = f["fen"][index]
        if isinstance(fen, bytes):
            fen_s = fen.decode("utf-8")
        else:
            fen_s = str(fen)
        out: dict[str, torch.Tensor | str] = {
            "fen": fen_s,
            "side_to_move": torch.tensor(int(f["side_to_move"][index]), dtype=torch.long),
            "elo_to_move": torch.tensor(int(f["elo_to_move"][index]), dtype=torch.long),
            "from_sq": torch.tensor(int(f["from_sq"][index]), dtype=torch.long),
            "to_sq": torch.tensor(int(f["to_sq"][index]), dtype=torch.long),
            "promotion": torch.tensor(int(f["promotion"][index]), dtype=torch.long),
            "stratum_index": torch.tensor(int(f["stratum_index"][index]), dtype=torch.long),
        }
        if "source_plan_index" in f:
            out["source_plan_index"] = torch.tensor(
                int(f["source_plan_index"][index]), dtype=torch.long
            )
        return out

    def numpy_row(self, index: int) -> dict[str, np.ndarray | str]:
        f = self._file()
        fen = f["fen"][index]
        if isinstance(fen, bytes):
            fen_s = fen.decode("utf-8")
        else:
            fen_s = str(fen)
        out: dict[str, np.ndarray | str] = {
            "fen": fen_s,
            "side_to_move": np.array(f["side_to_move"][index]),
            "elo_to_move": np.array(f["elo_to_move"][index]),
            "from_sq": np.array(f["from_sq"][index]),
            "to_sq": np.array(f["to_sq"][index]),
            "promotion": np.array(f["promotion"][index]),
            "stratum_index": np.array(f["stratum_index"][index]),
        }
        if "source_plan_index" in f:
            out["source_plan_index"] = np.array(f["source_plan_index"][index])
        return out
