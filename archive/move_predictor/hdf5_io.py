"""Shared HDF5 layout for move predictor datasets."""

from __future__ import annotations

import random
from pathlib import Path

import h5py
import numpy as np


def shuffle_columns(
    triple_from: list[int],
    triple_to: list[int],
    triple_prom: list[int],
    rng: random.Random,
) -> tuple[list[int], list[int], list[int], int]:
    perm = [0, 1, 2]
    rng.shuffle(perm)
    sf = [triple_from[perm[i]] for i in range(3)]
    st = [triple_to[perm[i]] for i in range(3)]
    sp = [triple_prom[perm[i]] for i in range(3)]
    label = perm.index(0)
    return sf, st, sp, label


def flush_move_h5(
    h5: h5py.File,
    *,
    embedding_dim: int,
    history_n: int,
    cur_list: list[np.ndarray],
    hist_white_list: list[np.ndarray],
    hist_black_list: list[np.ndarray],
    hlen_white_list: list[int],
    hlen_black_list: list[int],
    side_to_move_list: list[int],
    from_list: list[np.ndarray],
    to_list: list[np.ndarray],
    prom_list: list[np.ndarray],
    label_list: list[int],
    fen_list: list[str],
    next_idx: int,
) -> int:
    if not cur_list:
        return next_idx
    n = len(cur_list)
    new_size = next_idx + n
    cur_ds = h5["cur_emb"]
    hw_ds = h5["hist_white_emb"]
    hb_ds = h5["hist_black_emb"]
    hlw_ds = h5["hist_white_len"]
    hlb_ds = h5["hist_black_len"]
    turn_ds = h5["side_to_move"]
    from_ds = h5["from_sq"]
    to_ds = h5["to_sq"]
    prom_ds = h5["promotion"]
    label_ds = h5["label"]
    fen_ds = h5["fen"]

    for ds in (
        cur_ds,
        hw_ds,
        hb_ds,
        hlw_ds,
        hlb_ds,
        turn_ds,
        from_ds,
        to_ds,
        prom_ds,
        label_ds,
    ):
        ds.resize(new_size, axis=0)
    fen_ds.resize((new_size,))

    cur_ds[next_idx:new_size] = np.stack(cur_list, axis=0)
    hw_ds[next_idx:new_size] = np.stack(hist_white_list, axis=0)
    hb_ds[next_idx:new_size] = np.stack(hist_black_list, axis=0)
    hlw_ds[next_idx:new_size] = np.asarray(hlen_white_list, dtype=np.int32)
    hlb_ds[next_idx:new_size] = np.asarray(hlen_black_list, dtype=np.int32)
    turn_ds[next_idx:new_size] = np.asarray(side_to_move_list, dtype=np.uint8)
    from_ds[next_idx:new_size] = np.stack(from_list, axis=0)
    to_ds[next_idx:new_size] = np.stack(to_list, axis=0)
    prom_ds[next_idx:new_size] = np.stack(prom_list, axis=0)
    label_ds[next_idx:new_size] = np.asarray(label_list, dtype=np.uint8)
    fen_ds[next_idx:new_size] = fen_list
    return new_size


def ensure_move_h5(path: Path, embedding_dim: int, history_n: int) -> h5py.File:
    f = h5py.File(path, "a")
    if "cur_emb" not in f:
        f.create_dataset(
            "cur_emb",
            shape=(0, embedding_dim),
            maxshape=(None, embedding_dim),
            dtype=np.float32,
            chunks=(1, embedding_dim),
        )
        f.create_dataset(
            "hist_white_emb",
            shape=(0, history_n, embedding_dim),
            maxshape=(None, history_n, embedding_dim),
            dtype=np.float32,
        )
        f.create_dataset(
            "hist_black_emb",
            shape=(0, history_n, embedding_dim),
            maxshape=(None, history_n, embedding_dim),
            dtype=np.float32,
        )
        f.create_dataset("hist_white_len", shape=(0,), maxshape=(None,), dtype=np.int32)
        f.create_dataset("hist_black_len", shape=(0,), maxshape=(None,), dtype=np.int32)
        f.create_dataset("side_to_move", shape=(0,), maxshape=(None,), dtype=np.uint8)
        f.create_dataset("from_sq", shape=(0, 3), maxshape=(None, 3), dtype=np.uint8)
        f.create_dataset("to_sq", shape=(0, 3), maxshape=(None, 3), dtype=np.uint8)
        f.create_dataset("promotion", shape=(0, 3), maxshape=(None, 3), dtype=np.uint8)
        f.create_dataset("label", shape=(0,), maxshape=(None,), dtype=np.uint8)
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("fen", shape=(0,), maxshape=(None,), dtype=dt)
        f.attrs["embedding_dim"] = embedding_dim
        f.attrs["history_n"] = history_n
    return f
