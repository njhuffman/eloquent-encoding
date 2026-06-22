import torch
from style_policy.model import BasePolicy
from style_policy.onnx_export import build_export_modules

CFG = dict(d_model=256, n_layers=8, nhead=8, dim_feedforward=1024, dropout=0.0,
           head_hidden=512, elo_dim=32, n_elo_buckets=40)

def _board_tensor(b=2):
    # random but valid: one-hot piece planes on a few squares, plane 12 turn bit
    t = torch.zeros(b, 8, 8, 18)
    t[:, 0, 0, 3] = 1.0   # a rook on a1
    t[:, 7, 4, 11] = 1.0  # a black king-ish
    t[:, :, :, 12] = 1.0  # white to move
    return t

def test_export_wrappers_match_eager():
    policy = BasePolicy.from_config(CFG).eval()
    enc, fh, th = build_export_modules(policy)
    bt = _board_tensor()
    with torch.no_grad():
        _, squares_ref = policy.encoder(bt)
        squares = enc(bt)
        assert torch.allclose(squares, squares_ref, atol=1e-5)
        elo = torch.tensor([12, 18], dtype=torch.long)
        assert torch.allclose(fh(squares, elo), policy.from_head(squares, elo_idx=elo), atol=1e-5)
        fsq = torch.tensor([0, 4], dtype=torch.long)
        assert torch.allclose(th(squares, fsq, elo), policy.to_head(squares, fsq, elo_idx=elo), atol=1e-5)
