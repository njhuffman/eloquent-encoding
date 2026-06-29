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


def test_eval_row_metric_on_forced_moves(tmp_path):
    import numpy as np, h5py, torch
    from style_policy.model import BasePolicy
    from style_policy.band_head import BandHead, eval_band_head_row
    from style_policy.packed_codec import PACKED_BOARD_LEN
    arch = {"d_model": 32, "n_layers": 1, "nhead": 4, "dim_feedforward": 64,
            "dropout": 0.0, "head_hidden": 32, "elo_dim": 8, "n_elo_buckets": 40}
    m = BasePolicy.from_config(arch); ckpt = tmp_path / "enc.pt"
    torch.save({"model": m.state_dict(), "architecture": arch}, ckpt)
    # one legal from-square and one legal to-square == the human move => any head scores 100%
    n = 8; vp = tmp_path / "val.h5"
    packed = np.zeros((n, PACKED_BOARD_LEN), np.uint8); packed[:, 33] = 255
    with h5py.File(vp, "w") as f:
        f["packed_pre"] = packed
        f["elo_to_move"] = np.full(n, 1900, np.int16)
        f["from_sq"] = np.full(n, 12, np.uint8); f["to_sq"] = np.full(n, 28, np.uint8)
        f["from_legal_u64"] = np.full(n, np.uint64(1) << np.uint64(12), np.uint64)
        f["to_legal_u64"] = np.full(n, np.uint64(1) << np.uint64(28), np.uint64)
        f["promotion"] = np.zeros(n, np.uint8); f["opp_elo"] = np.full(n, 1500, np.int16)
        f["result"] = np.ones(n, np.int8)
    head = BandHead(32, 32)
    rows = eval_band_head_row(str(ckpt), head, str(vp), [1900], device="cpu", n=n)
    assert rows[1900]["count"] == n
    assert rows[1900]["spec"] == 100.0 and rows[1900]["shared"] == 100.0
