import numpy as np
import pytest
import h5py
import torch
from style_policy.dataset import PackedMoveDataset

N = 32  # number of rows in the synthetic fixture


@pytest.fixture
def h5_path(tmp_path):
    """Build a minimal h5 fixture with all columns PackedMoveDataset.__getitem__ reads."""
    rng = np.random.default_rng(42)
    path = tmp_path / "fixture.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("packed_pre",      data=rng.integers(0, 256, (N, 34), dtype=np.uint8))
        f.create_dataset("packed_post",     data=rng.integers(0, 256, (N, 34), dtype=np.uint8))
        # ensure from_sq always indexes a set bit in from_legal_u64
        from_legal = rng.integers(1, 2**63, N, dtype=np.uint64)
        from_sq = np.array(
            [rng.choice([b for b in range(64) if (int(v) >> b) & 1]) for v in from_legal],
            dtype=np.uint8,
        )
        f.create_dataset("from_legal_u64",  data=from_legal)
        f.create_dataset("to_legal_u64",    data=rng.integers(1, 2**63, N, dtype=np.uint64))
        f.create_dataset("elo_to_move",     data=rng.integers(800, 2800, N, dtype=np.int16))
        f.create_dataset("from_sq",         data=from_sq)
        f.create_dataset("to_sq",           data=rng.integers(0, 64, N, dtype=np.uint8))
        f.create_dataset("promotion",       data=np.zeros(N, dtype=np.uint8))
        # columns added by the wdl-value-head branch
        f.create_dataset("result",          data=rng.integers(0, 3, N, dtype=np.int8))
        f.create_dataset("opp_elo",         data=rng.integers(800, 2800, N, dtype=np.int16))
    return str(path)


def test_row_fields_and_shapes(h5_path):
    ds = PackedMoveDataset(h5_path, sample_n=16, seed=1)
    assert len(ds) == 16
    row = ds[0]
    assert row["packed_pre"].shape == (34,)
    assert row["from_sq"].dtype == torch.int64
    # ground-truth from_sq is a legal origin
    assert (int(row["from_legal_u64"]) >> int(row["from_sq"])) & 1 == 1


def test_collate_batches(h5_path):
    ds = PackedMoveDataset(h5_path, sample_n=8, seed=1)
    batch = PackedMoveDataset.collate([ds[i] for i in range(8)])
    assert batch["packed_pre"].shape == (8, 34)
    assert batch["from_sq"].shape == (8,)
