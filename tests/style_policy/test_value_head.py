import torch
from style_policy.value_head import WDLHead

def test_shapes_and_elo_conditioning():
    h = WDLHead(d_model=32, hidden=16, elo_dim=8, n_elo_buckets=40).eval()
    cls = torch.randn(5, 32)
    out = h(cls, elo_idx=torch.tensor([0, 1, 2, 3, 4]))
    assert out.shape == (5, 3)
    # null-elo path works and differs from a real bucket
    out_null = h(cls, elo_idx=None)
    assert out_null.shape == (5, 3)
    assert not torch.allclose(out, out_null)

def test_no_elo_variant():
    h = WDLHead(d_model=32, hidden=16, elo_dim=0, n_elo_buckets=0).eval()
    assert h(torch.randn(2, 32), elo_idx=None).shape == (2, 3)
