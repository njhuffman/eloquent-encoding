"""Smoke test for chess_jepa_v3 forward."""

from __future__ import annotations

import torch

from jepa3.architectures.chess_jepa_v3 import ChessJEPAV3, resolve_architecture_config


def test_chess_jepa_v3_forward_smoke() -> None:
    cfg = resolve_architecture_config(
        {
            "d_model": 64,
            "encoder_layers": 1,
            "nhead": 4,
            "dim_feedforward": 128,
            "dropout": 0.0,
            "use_cls": True,
            "jepa_square_embed_dim": 24,
            "predictor_hidden": 128,
            "predictor_depth": 2,
            "from_to_head_hidden": 96,
            "from_to_head_depth": 2,
        }
    )
    m = ChessJEPAV3(cfg)
    m.eval()
    b, h, w, c = 2, 8, 8, 18
    board = torch.randn(b, h, w, c)
    post = torch.randn(b, h, w, c)
    fs = torch.tensor([12, 55], dtype=torch.long)
    ts = torch.tensor([28, 56], dtype=torch.long)
    z = m.encode_online(board)
    assert z.shape == (b, cfg["d_model"])
    z2, z_hat = m.encode_online_with_jepa(board, fs, ts)
    assert z2.shape == z.shape
    assert z_hat.shape == (b, cfg["d_model"])
    z_pos = m.encode_target_global(post)
    assert z_pos.shape == (b, cfg["d_model"])
    lf = m.forward_from_logits(z)
    assert lf.shape == (b, 64)
    lt = m.forward_to_logits(z, fs)
    assert lt.shape == (b, 64)


def test_from_to_heads_backprop_into_full_z() -> None:
    cfg = resolve_architecture_config(
        {
            "d_model": 64,
            "encoder_layers": 1,
            "nhead": 4,
            "dim_feedforward": 128,
            "dropout": 0.0,
            "use_cls": True,
            "jepa_square_embed_dim": 24,
            "predictor_hidden": 128,
            "predictor_depth": 2,
            "from_to_head_hidden": 96,
            "from_to_head_depth": 2,
        }
    )
    m = ChessJEPAV3(cfg)
    z = torch.randn(2, cfg["d_model"], requires_grad=True)
    lf = m.forward_from_logits(z)
    lf.sum().backward()
    assert z.grad is not None
    assert z.grad.abs().max().item() > 0.0

    z2 = torch.randn(2, cfg["d_model"], requires_grad=True)
    lt = m.forward_to_logits(z2, torch.tensor([0, 1], dtype=torch.long))
    lt.sum().backward()
    assert z2.grad is not None
    assert z2.grad.abs().max().item() > 0.0
