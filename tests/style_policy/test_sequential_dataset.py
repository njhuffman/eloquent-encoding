import h5py
import numpy as np
from pathlib import Path
from style_policy.dataset import PackedMoveDataset


def _mk(path: Path, n: int) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("packed_pre", data=np.zeros((n, 34), np.uint8))
        f.create_dataset("elo_to_move", data=np.arange(1000, 1000 + n, dtype=np.int64))


def test_sequential_takes_first_n_in_order(tmp_path):
    p = tmp_path / "d.h5"; _mk(p, 100)
    ds = PackedMoveDataset(p, sample_n=10, seed=0, sequential=True)
    assert list(ds.indices) == list(range(10))


def test_random_mode_unchanged(tmp_path):
    p = tmp_path / "d.h5"; _mk(p, 100)
    ds = PackedMoveDataset(p, sample_n=10, seed=0, sequential=False)
    expected = np.sort(np.random.default_rng(0).choice(np.arange(100), size=10, replace=False))
    assert list(ds.indices) == list(expected)


def test_sequential_with_band_filter(tmp_path):
    p = tmp_path / "d.h5"; _mk(p, 100)  # elo 1000..1099 over indices 0..99
    ds = PackedMoveDataset(p, sample_n=5, seed=0, band=(1000, 1050), sequential=True)
    assert list(ds.indices) == list(range(5))  # band pool = 0..49; first 5 in order
