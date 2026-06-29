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


def test_train_band_head_smoke(tmp_path):
    import torch
    from style_policy.model import BasePolicy
    from style_policy.band_head import train_band_head, BandHead
    arch = {"d_model": 32, "n_layers": 1, "nhead": 4, "dim_feedforward": 64,
            "dropout": 0.0, "head_hidden": 32, "elo_dim": 8, "n_elo_buckets": 40}
    m = BasePolicy.from_config(arch)
    ckpt = tmp_path / "enc.pt"
    torch.save({"model": m.state_dict(), "architecture": arch}, ckpt)
    from tests.style_policy.synth_h5 import write_synth_h5
    h5 = write_synth_h5(tmp_path / "train.h5", elos=[1900]*256)
    out = tmp_path / "head.pt"
    head, meta = train_band_head(str(ckpt), 1900, str(h5), device="cpu",
                                 steps=5, batch_size=64, num_workers=0, out=str(out))
    assert meta["band"] == 1900 and meta["d_model"] == 32
    loaded = BandHead(meta["d_model"], meta["hidden"])
    loaded.load_state_dict(torch.load(out)["band_head"])  # clean load
