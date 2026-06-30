import numpy as np
import h5py
from dataset_generation.hdf5_io import PackedBatchWriter

def test_packed_writer_roundtrip(tmp_path):
    p = tmp_path / "x.h5"
    pre = np.arange(34, dtype=np.uint8)
    absent_hist = np.array([-1, -1, -1, -1], dtype=np.int8)  # -1 represents 255 as absent marker
    with PackedBatchWriter(p, batch_size=2) as w:
        for i in range(3):
            w.append_row(packed_pre=pre + i, from_legal_u64=(1 << 63) | 1, to_legal_u64=5,
                         from_sq=12, to_sq=28, promotion=0, elo_to_move=1500, opp_elo=1600, result=2,
                         hist_from=absent_hist, hist_to=absent_hist, hist_cap=np.array([0, 0, 0, 0], dtype=np.int8))
    with h5py.File(p, "r") as f:
        assert f["packed_pre"].shape == (3, 34) and f["packed_pre"].dtype == np.uint8
        assert int(f["result"][0]) == 2 and str(f["result"].dtype) == "int8"
        assert int(f["opp_elo"][1]) == 1600
        assert np.uint64(f["from_legal_u64"][2]) == np.uint64((1 << 63) | 1)  # bit 63 preserved
        assert list(f["packed_pre"][1]) == list((pre + 1))
