# tests/dataset_generation/test_shuffle_merge.py
import h5py
import numpy as np
import pytest
from pathlib import Path
from dataset_generation.shuffle_merge import shuffle_merge


def _fake_shard(path: Path, a_values: np.ndarray) -> None:
    """Two datasets with a known per-row relationship: b == a*10, vec == [a, a+1]."""
    with h5py.File(path, "w") as f:
        f.create_dataset("a", data=a_values.astype(np.int64))
        f.create_dataset("b", data=(a_values * 10).astype(np.int64))
        f.create_dataset("vec", data=np.stack([a_values, a_values + 1], axis=1).astype(np.int64))


def test_shuffle_merge_complete_aligned_deterministic(tmp_path):
    s0 = tmp_path / "s0.h5"; _fake_shard(s0, np.arange(0, 10))
    s1 = tmp_path / "s1.h5"; _fake_shard(s1, np.arange(10, 25))
    out = shuffle_merge([s0, s1], tmp_path / "merged.h5", seed=7)
    with h5py.File(out, "r") as f:
        a = f["a"][:]; b = f["b"][:]; vec = f["vec"][:]
    assert len(a) == 25                                   # completeness (count)
    assert sorted(a.tolist()) == list(range(25))          # completeness (values)
    assert np.array_equal(b, a * 10)                      # alignment a<->b
    assert np.array_equal(vec[:, 0], a) and np.array_equal(vec[:, 1], a + 1)  # alignment a<->vec
    assert not np.array_equal(a, np.sort(a))              # actually shuffled (seed 7, n=25)


def test_shuffle_merge_seed_reproducible(tmp_path):
    s0 = tmp_path / "s0.h5"; _fake_shard(s0, np.arange(0, 10))
    o1 = shuffle_merge([s0], tmp_path / "m1.h5", seed=3)
    o2 = shuffle_merge([s0], tmp_path / "m2.h5", seed=3)
    with h5py.File(o1, "r") as f1, h5py.File(o2, "r") as f2:
        assert np.array_equal(f1["a"][:], f2["a"][:])


def test_shuffle_merge_rejects_mismatched_datasets(tmp_path):
    s0 = tmp_path / "s0.h5"; _fake_shard(s0, np.arange(0, 5))
    with h5py.File(tmp_path / "s1.h5", "w") as f:
        f.create_dataset("a", data=np.arange(5).astype(np.int64))  # missing b, vec
    with pytest.raises(AssertionError):
        shuffle_merge([s0, tmp_path / "s1.h5"], tmp_path / "m.h5", seed=1)
