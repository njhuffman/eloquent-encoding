"""chess_jepa_v4: aux on full global embedding."""

from __future__ import annotations

import torch

from jepa3.architectures.chess_jepa_v4 import ChessJEPAV4Builder


def test_aux_backward_has_encoder_grad() -> None:
    torch.manual_seed(0)
    cfg = {
        "d_model": 64,
        "encoder_layers": 1,
        "nhead": 4,
        "dim_feedforward": 128,
        "dropout": 0.0,
        "jepa_square_embed_dim": 16,
        "predictor_hidden": 128,
        "predictor_depth": 2,
        "from_to_head_hidden": 64,
        "from_to_head_depth": 2,
        "aux_board_recon_hidden": 32,
        "aux_meta_hidden": 32,
    }
    model = ChessJEPAV4Builder.build(cfg)
    model.train()
    b = 2
    board = torch.zeros(b, 8, 8, 18)
    board[:, 0, 0, 0] = 1.0
    board[:, 0, 0, 12] = 1.0
    fs = torch.zeros(b, dtype=torch.long)
    ts = torch.ones(b, dtype=torch.long)
    z, _ = model.encode_online_with_jepa(board, fs, ts)
    out = model.forward_prefix_aux_losses(board, z, compute_board_recon=True, compute_meta=True)
    loss = out["aux_board_recon"] + out["aux_meta"]
    loss.backward()
    enc_grad = any(
        p.grad is not None and p.grad.abs().max().item() > 0 for p in model.encoder_online.parameters()
    )
    assert enc_grad
    br_grad = any(
        p.grad is not None and p.grad.abs().max().item() > 0 for p in model.aux_board_recon.parameters()
    )
    assert br_grad
    meta_grad = any(
        p.grad is not None and p.grad.abs().max().item() > 0 for p in model.aux_meta.parameters()
    )
    assert meta_grad


def test_from_logits_backprop_uses_all_dims() -> None:
    base = {
        "d_model": 128,
        "encoder_layers": 1,
        "nhead": 4,
        "dim_feedforward": 256,
        "dropout": 0.0,
        "jepa_square_embed_dim": 16,
        "predictor_hidden": 128,
        "predictor_depth": 2,
        "from_to_head_hidden": 64,
        "from_to_head_depth": 2,
        "aux_board_recon_hidden": 32,
        "aux_meta_hidden": 32,
    }
    z = torch.randn(1, 128, requires_grad=True)
    m = ChessJEPAV4Builder.build(base)
    m.train()
    logits = m.forward_from_logits(z)
    logits.sum().backward()
    assert z.grad is not None
    assert z.grad.abs().max().item() > 0.0
