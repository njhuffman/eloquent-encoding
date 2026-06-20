"""HDF5 dataset for move predictor: one file handle per worker, batched slice reads."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def h5_num_rows(h5_path: str | Path) -> int:
    with h5py.File(h5_path, "r") as f:
        return int(f["cur_emb"].shape[0])


class MovePredictorH5Dataset(Dataset):
    """
    One ``h5py`` handle per worker (``swmr=True``), opened lazily.

    Training uses :class:`torch.utils.data.BatchSampler`; the loader calls
    ``__getitems__(indices)`` with the full index list per batch. Indices are
    sorted and merged into contiguous HDF5 row slices so each batch does a few
    large reads instead of one read per row. Rows are reordered to match the
    original index order (required when shuffling).

    Each sample includes separate white / black history streams and ``side_to_move``
    (0=white, 1=black) for the current position.
    """

    def __init__(self, file_path: str | Path):
        self.file_path = str(file_path)
        self._archive: h5py.File | None = None
        with h5py.File(self.file_path, "r") as f:
            self.length = int(f["cur_emb"].shape[0])

    def __len__(self) -> int:
        return self.length

    def _ensure_open(self) -> h5py.File:
        if self._archive is None:
            self._archive = h5py.File(self.file_path, "r", swmr=True)
        return self._archive

    def __getitem__(self, index: int):
        return self.__getitems__([index])[0]

    def __getitems__(self, indices: list[int]) -> list:
        if not indices:
            return []
        f = self._ensure_open()
        idx = np.asarray(indices, dtype=np.int64)
        order = np.argsort(idx, kind="mergesort")
        sorted_idx = idx[order]
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        b = len(idx)

        sorted_cur: np.ndarray | None = None
        sorted_hw: np.ndarray | None = None
        sorted_hb: np.ndarray | None = None
        sorted_lw: np.ndarray | None = None
        sorted_lb: np.ndarray | None = None
        sorted_turn: np.ndarray | None = None
        sorted_from: np.ndarray | None = None
        sorted_to: np.ndarray | None = None
        sorted_y: np.ndarray | None = None

        pos = 0
        while pos < b:
            start_row = int(sorted_idx[pos])
            q = pos + 1
            while q < b and int(sorted_idx[q]) == int(sorted_idx[q - 1]) + 1:
                q += 1
            end_row = int(sorted_idx[q - 1]) + 1

            cur_b = np.asarray(f["cur_emb"][start_row:end_row], dtype=np.float32)
            hw_b = np.asarray(f["hist_white_emb"][start_row:end_row], dtype=np.float32)
            hb_b = np.asarray(f["hist_black_emb"][start_row:end_row], dtype=np.float32)
            lw_b = np.asarray(f["hist_white_len"][start_row:end_row], dtype=np.int64)
            lb_b = np.asarray(f["hist_black_len"][start_row:end_row], dtype=np.int64)
            turn_b = np.asarray(f["side_to_move"][start_row:end_row], dtype=np.int64)
            from_b = np.asarray(f["from_sq"][start_row:end_row], dtype=np.int64)
            to_b = np.asarray(f["to_sq"][start_row:end_row], dtype=np.int64)
            y_b = np.asarray(f["label"][start_row:end_row], dtype=np.int64)

            if sorted_cur is None:
                emb_d = cur_b.shape[-1]
                hn, ed = hw_b.shape[1], hw_b.shape[2]
                assert ed == emb_d and hb_b.shape[1:] == (hn, emb_d)
                sorted_cur = np.empty((b, emb_d), dtype=np.float32)
                sorted_hw = np.empty((b, hn, emb_d), dtype=np.float32)
                sorted_hb = np.empty((b, hn, emb_d), dtype=np.float32)
                sorted_lw = np.empty((b,), dtype=np.int64)
                sorted_lb = np.empty((b,), dtype=np.int64)
                sorted_turn = np.empty((b,), dtype=np.int64)
                sorted_from = np.empty((b,) + from_b.shape[1:], dtype=np.int64)
                sorted_to = np.empty((b,) + to_b.shape[1:], dtype=np.int64)
                sorted_y = np.empty((b,), dtype=np.int64)

            sorted_cur[pos:q] = cur_b
            sorted_hw[pos:q] = hw_b
            sorted_hb[pos:q] = hb_b
            sorted_lw[pos:q] = lw_b
            sorted_lb[pos:q] = lb_b
            sorted_turn[pos:q] = turn_b
            sorted_from[pos:q] = from_b
            sorted_to[pos:q] = to_b
            sorted_y[pos:q] = y_b
            pos = q

        assert sorted_cur is not None
        cur_r = sorted_cur[inv]
        hw_r = sorted_hw[inv]
        hb_r = sorted_hb[inv]
        lw_r = sorted_lw[inv]
        lb_r = sorted_lb[inv]
        turn_r = sorted_turn[inv]
        from_r = sorted_from[inv]
        to_r = sorted_to[inv]
        y_r = sorted_y[inv]

        cur_t = torch.from_numpy(np.ascontiguousarray(cur_r))
        hw_t = torch.from_numpy(np.ascontiguousarray(hw_r))
        hb_t = torch.from_numpy(np.ascontiguousarray(hb_r))
        lw_t = torch.from_numpy(lw_r.copy())
        lb_t = torch.from_numpy(lb_r.copy())
        turn_t = torch.from_numpy(turn_r.copy())
        from_t = torch.from_numpy(np.ascontiguousarray(from_r))
        to_t = torch.from_numpy(np.ascontiguousarray(to_r))
        y_t = torch.from_numpy(y_r.copy())

        return [
            (cur_t[i], hw_t[i], hb_t[i], lw_t[i], lb_t[i], turn_t[i], from_t[i], to_t[i], y_t[i])
            for i in range(b)
        ]
