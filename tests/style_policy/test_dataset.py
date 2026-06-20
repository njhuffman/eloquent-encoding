import torch
from style_policy.dataset import PackedMoveDataset

H5 = "/mnt/eloquence_bulk/databases/j3_training_1M.h5"


def test_row_fields_and_shapes():
    ds = PackedMoveDataset(H5, sample_n=16, seed=1)
    assert len(ds) == 16
    row = ds[0]
    assert row["packed_pre"].shape == (34,)
    assert row["from_sq"].dtype == torch.int64
    # ground-truth from_sq is a legal origin
    assert (int(row["from_legal_u64"]) >> int(row["from_sq"])) & 1 == 1


def test_collate_batches():
    ds = PackedMoveDataset(H5, sample_n=8, seed=1)
    batch = PackedMoveDataset.collate([ds[i] for i in range(8)])
    assert batch["packed_pre"].shape == (8, 34)
    assert batch["from_sq"].shape == (8,)
