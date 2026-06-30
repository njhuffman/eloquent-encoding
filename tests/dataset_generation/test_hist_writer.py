import numpy as np
import h5py
from dataset_generation.hdf5_io import PackedBatchWriter


def test_hist_columns_roundtrip(tmp_path):
    """Test that hist_from, hist_to, hist_cap columns are created, flushed, and round-trip correctly."""
    p = tmp_path / "x.h5"
    pre = np.arange(34, dtype=np.uint8)

    # Write 3 rows with known history
    # Row 0: normal history
    hist_from_0 = np.array([12, 28, 4, 60], dtype=np.int8)
    hist_to_0 = np.array([28, 36, 16, 45], dtype=np.int8)
    hist_cap_0 = np.array([0, 1, 2, 0], dtype=np.int8)

    # Row 1: partial history (absent plies marked with -1, which represents 255 when cast as uint8)
    hist_from_1 = np.array([32, 52, -1, -1], dtype=np.int8)
    hist_to_1 = np.array([40, 60, -1, -1], dtype=np.int8)
    hist_cap_1 = np.array([3, 4, 0, 0], dtype=np.int8)

    # Row 2: all-absent history (all -1 in from/to, 0 in cap)
    hist_from_2 = np.array([-1, -1, -1, -1], dtype=np.int8)
    hist_to_2 = np.array([-1, -1, -1, -1], dtype=np.int8)
    hist_cap_2 = np.array([0, 0, 0, 0], dtype=np.int8)

    with PackedBatchWriter(p, batch_size=10) as w:
        w.append_row(
            packed_pre=pre,
            from_legal_u64=1, to_legal_u64=5,
            from_sq=12, to_sq=28, promotion=0,
            elo_to_move=1500, opp_elo=1600, result=1,
            hist_from=hist_from_0, hist_to=hist_to_0, hist_cap=hist_cap_0,
        )
        w.append_row(
            packed_pre=pre + 1,
            from_legal_u64=2, to_legal_u64=6,
            from_sq=32, to_sq=40, promotion=0,
            elo_to_move=1600, opp_elo=1700, result=0,
            hist_from=hist_from_1, hist_to=hist_to_1, hist_cap=hist_cap_1,
        )
        w.append_row(
            packed_pre=pre + 2,
            from_legal_u64=3, to_legal_u64=7,
            from_sq=24, to_sq=32, promotion=0,
            elo_to_move=1700, opp_elo=1800, result=-1,
            hist_from=hist_from_2, hist_to=hist_to_2, hist_cap=hist_cap_2,
        )

    # Reopen and verify
    with h5py.File(p, "r") as f:
        # Check shapes
        assert f["hist_from"].shape == (3, 4), f"Expected (3,4), got {f['hist_from'].shape}"
        assert f["hist_to"].shape == (3, 4), f"Expected (3,4), got {f['hist_to'].shape}"
        assert f["hist_cap"].shape == (3, 4), f"Expected (3,4), got {f['hist_cap'].shape}"

        # Check dtypes
        assert f["hist_from"].dtype == np.int8
        assert f["hist_to"].dtype == np.int8
        assert f["hist_cap"].dtype == np.int8

        # Check row-trip values for row 0
        assert np.array_equal(f["hist_from"][0], hist_from_0)
        assert np.array_equal(f["hist_to"][0], hist_to_0)
        assert np.array_equal(f["hist_cap"][0], hist_cap_0)

        # Check round-trip values for row 1
        assert np.array_equal(f["hist_from"][1], hist_from_1)
        assert np.array_equal(f["hist_to"][1], hist_to_1)
        assert np.array_equal(f["hist_cap"][1], hist_cap_1)

        # Check round-trip values for row 2 (all-absent)
        assert np.array_equal(f["hist_from"][2], hist_from_2)
        assert np.array_equal(f["hist_to"][2], hist_to_2)
        assert np.array_equal(f["hist_cap"][2], hist_cap_2)
