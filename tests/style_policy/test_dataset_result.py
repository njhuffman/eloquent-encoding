import numpy as np, h5py, torch
from dataset_generation.hdf5_io import PackedBatchWriter
from style_policy.dataset import PackedMoveDataset

def _tiny(path):
    with PackedBatchWriter(path, batch_size=4) as w:
        absent_hist = np.array([-1, -1, -1, -1], dtype=np.int8)
        for i in range(4):
            w.append_row(packed_pre=np.zeros(34, np.uint8), from_legal_u64=1, to_legal_u64=1,
                         from_sq=0, to_sq=1, promotion=0, elo_to_move=1500, opp_elo=1400, result=i % 3,
                         hist_from=absent_hist, hist_to=absent_hist, hist_cap=np.array([0, 0, 0, 0], dtype=np.int8))

def test_dataset_exposes_result(tmp_path):
    p = tmp_path / "d.h5"; _tiny(p)
    ds = PackedMoveDataset(p)
    item = ds[2]
    assert "result" in item and int(item["result"]) == 2
    batch = PackedMoveDataset.collate([ds[0], ds[1], ds[2]])
    assert batch["result"].tolist() == [0, 1, 2]
