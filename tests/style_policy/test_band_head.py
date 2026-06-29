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
    # Drop value-head keys to mimic a policy-only checkpoint (e.g. base_64M); train must tolerate it.
    sd = {k: v for k, v in m.state_dict().items() if not k.startswith("value_head.")}
    torch.save({"model": sd, "architecture": arch}, ckpt)
    from tests.style_policy.synth_h5 import write_synth_h5
    h5 = write_synth_h5(tmp_path / "train.h5", elos=[1900]*256)
    out = tmp_path / "head.pt"
    head, meta = train_band_head(str(ckpt), 1900, str(h5), device="cpu",
                                 steps=5, batch_size=64, num_workers=0, out=str(out))
    assert meta["band"] == 1900 and meta["d_model"] == 32
    loaded = BandHead(meta["d_model"], meta["hidden"])
    loaded.load_state_dict(torch.load(out)["band_head"])  # clean load


def test_eval_row_metric_on_forced_moves(tmp_path):
    import numpy as np, h5py, torch, chess
    from style_policy.model import BasePolicy
    from style_policy.band_head import BandHead, eval_band_head_row
    from style_policy.board_encode import board_to_packed
    arch = {"d_model": 32, "n_layers": 1, "nhead": 4, "dim_feedforward": 64,
            "dropout": 0.0, "head_hidden": 32, "elo_dim": 8, "n_elo_buckets": 40}
    m = BasePolicy.from_config(arch); ckpt = tmp_path / "enc.pt"
    torch.save({"model": m.state_dict(), "architecture": arch}, ckpt)
    # A real position whose ONLY legal move is Kh8-g7. The eval derives legality from the
    # reconstructed board (legal_from / legal_to of the PREDICTED from-square), so with a single
    # legal move both the band head and the shared head must pick it -> 100% regardless of weights.
    # K+B+N vs K (sufficient material, so not "game over"); g8 attacked by Ba2, h7 by Nf8, h8 safe.
    board = chess.Board("5N1k/8/8/8/8/8/B7/K7 b - - 0 1")
    assert not board.is_game_over()
    assert [mv.uci() for mv in board.legal_moves] == ["h8g7"]  # guards the single-legal-move property
    row = np.asarray(board_to_packed(board), np.uint8)
    n = 4; vp = tmp_path / "val.h5"
    with h5py.File(vp, "w") as f:
        f["packed_pre"] = np.tile(row, (n, 1))
        f["elo_to_move"] = np.full(n, 1900, np.int16)
        f["from_sq"] = np.full(n, chess.H8, np.uint8)   # 63
        f["to_sq"] = np.full(n, chess.G7, np.uint8)     # 54
    head = BandHead(32, 32)
    rows = eval_band_head_row(str(ckpt), head, str(vp), [1900], device="cpu", n=n)
    assert rows[1900]["count"] == n
    assert rows[1900]["spec"] == 100.0 and rows[1900]["shared"] == 100.0
