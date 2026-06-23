import numpy as np, h5py, torch
from dataset_generation.hdf5_io import PackedBatchWriter
from style_policy.dataset import PackedMoveDataset
from style_policy.model import BasePolicy
from style_policy.training_loop import _step_loss

CFG = dict(d_model=32, n_layers=2, nhead=4, dim_feedforward=64, dropout=0.0,
           head_hidden=16, elo_dim=8, n_elo_buckets=40)

def _make(path, n=8):
    with PackedBatchWriter(path, batch_size=n) as w:
        for i in range(n):
            pre = np.zeros(34, np.uint8); pre[32] = 1
            w.append_row(packed_pre=pre, from_legal_u64=(1 << 1), to_legal_u64=(1 << 2),
                         from_sq=1, to_sq=2, promotion=0, elo_to_move=1500, opp_elo=1500, result=i % 3)

def test_step_loss_includes_value_term(tmp_path):
    p = tmp_path / "t.h5"; _make(p)
    ds = PackedMoveDataset(p)
    batch = PackedMoveDataset.collate([ds[i] for i in range(len(ds))])
    model = BasePolicy.from_config(CFG)
    loss, m = _step_loss(model, batch, "cpu", 40, 0.0)
    assert torch.isfinite(loss)
    assert "wdl_ce" in m and "wdl_acc" in m
    # with value_loss_weight default 1.0, total > from_ce + to_ce (value term is positive here)
    assert loss.item() > m["from_ce"] + m["to_ce"] - 1e-6
