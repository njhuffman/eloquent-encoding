"""Tests for optional last-move history columns in PackedMoveDataset (Task A1)."""
import numpy as np
import pytest
import torch

from tests.style_policy.synth_h5 import write_synth_h5
from style_policy.dataset import PackedMoveDataset


def test_dataset_without_hist_returns_absent_sentinels(tmp_path):
    """h5 with no hist_* datasets → __getitem__ returns all-absent tensors."""
    h5 = tmp_path / "no_hist.h5"
    write_synth_h5(h5, elos=[1500, 1600, 1700])

    ds = PackedMoveDataset(h5)
    item = ds[0]

    assert "hist_from" in item
    assert "hist_to" in item
    assert "hist_cap" in item

    assert item["hist_from"].shape == (4,)
    assert item["hist_to"].shape == (4,)
    assert item["hist_cap"].shape == (4,)

    assert item["hist_from"].dtype == torch.int64
    assert item["hist_to"].dtype == torch.int64
    assert item["hist_cap"].dtype == torch.int64

    assert (item["hist_from"] == -1).all(), f"expected all -1, got {item['hist_from']}"
    assert (item["hist_to"] == -1).all(), f"expected all -1, got {item['hist_to']}"
    assert (item["hist_cap"] == 0).all(), f"expected all 0, got {item['hist_cap']}"


def test_dataset_with_hist_returns_stored_values(tmp_path):
    """h5 WITH hist_* datasets → __getitem__ returns the stored (4,) values."""
    h5 = tmp_path / "with_hist.h5"
    elos = [1500, 1600, 1700]
    write_synth_h5(h5, elos=elos, with_hist=True, seed=42)

    ds = PackedMoveDataset(h5)

    import h5py
    with h5py.File(h5, "r") as f:
        expected_from = f["hist_from"][0].astype(np.int64)
        expected_to = f["hist_to"][0].astype(np.int64)
        expected_cap = f["hist_cap"][0].astype(np.int64)

    item = ds[0]

    assert item["hist_from"].shape == (4,)
    assert item["hist_to"].shape == (4,)
    assert item["hist_cap"].shape == (4,)

    assert item["hist_from"].dtype == torch.int64
    assert item["hist_to"].dtype == torch.int64
    assert item["hist_cap"].dtype == torch.int64

    assert (item["hist_from"].numpy() == expected_from).all(), \
        f"hist_from mismatch: got {item['hist_from']}, expected {expected_from}"
    assert (item["hist_to"].numpy() == expected_to).all(), \
        f"hist_to mismatch: got {item['hist_to']}, expected {expected_to}"
    assert (item["hist_cap"].numpy() == expected_cap).all(), \
        f"hist_cap mismatch: got {item['hist_cap']}, expected {expected_cap}"


def test_dataset_with_hist_all_rows(tmp_path):
    """All rows from a with-hist dataset return (4,) tensors matching stored data."""
    h5 = tmp_path / "with_hist_all.h5"
    elos = [1500, 1600, 1700, 1800]
    write_synth_h5(h5, elos=elos, with_hist=True, seed=7)

    ds = PackedMoveDataset(h5)

    import h5py
    with h5py.File(h5, "r") as f:
        stored_from = f["hist_from"][:].astype(np.int64)
        stored_to = f["hist_to"][:].astype(np.int64)
        stored_cap = f["hist_cap"][:].astype(np.int64)

    for i in range(len(ds)):
        item = ds[i]
        assert (item["hist_from"].numpy() == stored_from[i]).all()
        assert (item["hist_to"].numpy() == stored_to[i]).all()
        assert (item["hist_cap"].numpy() == stored_cap[i]).all()


def test_hist_absent_sentinel_values_mixed(tmp_path):
    """with_hist=True can store -1 sentinels; they round-trip correctly."""
    import h5py
    h5 = tmp_path / "mixed_hist.h5"
    # Write manually: row 0 fully absent, row 1 partial
    write_synth_h5(h5, elos=[1500, 1600])
    with h5py.File(h5, "a") as f:
        hist_from = np.array([[-1, -1, -1, -1], [12, 20, -1, -1]], dtype=np.int8)
        hist_to   = np.array([[-1, -1, -1, -1], [28, 35, -1, -1]], dtype=np.int8)
        hist_cap  = np.array([[0, 0, 0, 0],     [1, 0,  0,  0]], dtype=np.int8)
        f.create_dataset("hist_from", data=hist_from)
        f.create_dataset("hist_to",   data=hist_to)
        f.create_dataset("hist_cap",  data=hist_cap)

    ds = PackedMoveDataset(h5)

    item0 = ds[0]
    assert (item0["hist_from"] == -1).all()
    assert (item0["hist_to"]   == -1).all()
    assert (item0["hist_cap"]  ==  0).all()

    item1 = ds[1]
    assert item1["hist_from"][0].item() == 12
    assert item1["hist_to"][0].item()   == 28
    assert item1["hist_cap"][0].item()  == 1
    assert item1["hist_from"][2].item() == -1
    assert item1["hist_to"][2].item()   == -1


def test_collate_stacks_hist(tmp_path):
    """collate produces (B,4) tensors for hist columns."""
    h5 = tmp_path / "collate_hist.h5"
    write_synth_h5(h5, elos=[1500, 1600, 1700], with_hist=True, seed=0)
    ds = PackedMoveDataset(h5)

    batch = [ds[i] for i in range(3)]
    collated = PackedMoveDataset.collate(batch)

    assert collated["hist_from"].shape == (3, 4)
    assert collated["hist_to"].shape   == (3, 4)
    assert collated["hist_cap"].shape  == (3, 4)


def test_collate_stacks_hist_absent(tmp_path):
    """collate on a no-hist dataset produces (B,4) absent tensors."""
    h5 = tmp_path / "collate_no_hist.h5"
    write_synth_h5(h5, elos=[1500, 1600])
    ds = PackedMoveDataset(h5)

    batch = [ds[i] for i in range(2)]
    collated = PackedMoveDataset.collate(batch)

    assert collated["hist_from"].shape == (2, 4)
    assert (collated["hist_from"] == -1).all()
    assert (collated["hist_cap"]  ==  0).all()
