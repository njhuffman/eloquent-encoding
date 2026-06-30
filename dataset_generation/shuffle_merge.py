# dataset_generation/shuffle_merge.py
"""Concatenate band-shard h5 files and apply one seeded global permutation, producing a
training-ready shuffled file. Processes one dataset at a time to bound peak RAM (~2x the
largest dataset). Row alignment across datasets is guaranteed by the single shared perm."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


_WRITE_CHUNK = 1_000_000  # rows per permuted-write block (bounds the temp copy size)


def shuffle_merge(shard_paths, out_path, *, seed: int) -> Path:
    shard_paths = [Path(p) for p in shard_paths]
    if not shard_paths:
        raise ValueError("no shard paths given")
    with h5py.File(shard_paths[0], "r") as f0:
        names = sorted(f0.keys())
        specs = {k: (f0[k].dtype, tuple(f0[k].shape[1:])) for k in names}

    lengths: list[int] = []
    for p in shard_paths:
        with h5py.File(p, "r") as f:
            assert sorted(f.keys()) == names, f"dataset name mismatch in {p}"
            lengths.append(int(f[names[0]].shape[0]))
    n = sum(lengths)
    perm = np.random.default_rng(seed).permutation(n)

    out_path = Path(out_path)
    with h5py.File(out_path, "w") as out:
        for k in names:
            dt, tail = specs[k]
            # Hold one whole column (buf), then write its permuted form in chunks so we
            # never materialize a second full copy -> peak RAM ~= 1x the largest column
            # (plus the perm array), not 2x.
            buf = np.empty((n, *tail), dtype=dt)
            off = 0
            for p, ln in zip(shard_paths, lengths):
                with h5py.File(p, "r") as f:
                    buf[off:off + ln] = f[k][:]
                off += ln
            dset = out.create_dataset(k, shape=(n, *tail), dtype=dt)
            for i in range(0, n, _WRITE_CHUNK):
                dset[i:i + _WRITE_CHUNK] = buf[perm[i:i + _WRITE_CHUNK]]
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Seeded shuffle-merge of band-shard h5 files.")
    ap.add_argument("--shards", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(shuffle_merge(a.shards, a.out, seed=a.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
