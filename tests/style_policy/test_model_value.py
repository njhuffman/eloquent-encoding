import torch
from style_policy.model import BasePolicy

CFG = dict(d_model=64, n_layers=2, nhead=4, dim_feedforward=128, dropout=0.0,
           head_hidden=32, elo_dim=8, n_elo_buckets=40)

def _packed(b=3):
    t = torch.zeros(b, 34, dtype=torch.uint8)
    t[:, 32] = 1  # meta: white to move (bit 0)
    return t

def test_forward_value_shape_and_policy_returns_value():
    m = BasePolicy.from_config(CFG).eval()
    pk = _packed()
    elo = torch.tensor([10, 12, 14])
    v = m.forward_value(pk, elo_idx=elo)
    assert v.shape == (3, 3)
    fl, fm, tl, tm, v2 = m.forward_policy(pk, torch.zeros(3, dtype=torch.long),
                                          torch.ones(3, dtype=torch.int64), torch.ones(3, dtype=torch.int64),
                                          elo_idx=elo)
    assert v2.shape == (3, 3)
    assert torch.allclose(v, v2, atol=1e-5)  # same value from both paths (encode-once)
