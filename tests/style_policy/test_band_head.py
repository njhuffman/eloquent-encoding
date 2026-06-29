import torch
from style_policy.band_head import BandHead

def test_band_head_shapes_and_unconditioned():
    h = BandHead(d_model=32, hidden=16)
    sq = torch.randn(4, 64, 32)
    fl = h.from_logits(sq)
    assert fl.shape == (4, 64) and torch.isfinite(fl).all()
    from_sq = torch.zeros(4, dtype=torch.long)
    tl = h.to_logits(sq, from_sq)
    assert tl.shape == (4, 64) and torch.isfinite(tl).all()
    # unconditioned: no elo embedding anywhere
    assert not any("elo_emb" in n for n, _ in h.named_parameters())
    assert sum(p.numel() for p in h.parameters()) > 0
