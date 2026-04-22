"""chess_jepa_v4: prefix blend, probe gradients."""

from __future__ import annotations

import torch

from jepa3.architectures.chess_jepa_v4 import ChessJEPAV4Builder


def test_prefix_blend_matches_formula() -> None:
    m = ChessJEPAV4Builder.build({"d_model": 128, "predictor_prefix_dims": 64, "move_head_prefix_leak": 0.25})
    z = torch.randn(2, 128, requires_grad=True)
    out = m._z_global_for_move_heads(z, leak=0.25)
    pref = z[:, :64]
    expected = torch.cat([(1.0 - 0.25) * pref.detach() + 0.25 * pref, z[:, 64:]], dim=-1)
    assert torch.allclose(out, expected)


def test_probe_backward_no_encoder_grad() -> None:
    torch.manual_seed(0)
    cfg = {
        "d_model": 64,
        "encoder_layers": 1,
        "nhead": 4,
        "dim_feedforward": 128,
        "dropout": 0.0,
        "use_cls": True,
        "predictor_prefix_dims": 64,
        "jepa_square_embed_dim": 16,
        "predictor_hidden": 128,
        "predictor_depth": 2,
        "from_to_head_hidden": 64,
        "from_to_head_depth": 2,
        "probe_board_recon_hidden": 32,
        "probe_meta_hidden": 32,
        "move_head_prefix_leak": 0.0,
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
    aux = model.forward_aux_losses(board, z, compute_board_recon=True, compute_meta=True)
    loss = aux["probe_board_recon"] + aux["probe_meta"]
    loss.backward()
    for name, p in model.encoder_online.named_parameters():
        assert p.grad is None or p.grad.abs().max().item() == 0.0, name
    probe_grad = False
    for p in model.probe_board_recon.parameters():
        if p.grad is not None and p.grad.abs().max().item() > 0:
            probe_grad = True
    assert probe_grad
    meta_grad = any(
        p.grad is not None and p.grad.abs().max().item() > 0 for p in model.probe_meta.parameters()
    )
    assert meta_grad


def test_leak_zero_vs_one_encoder_grad_norm_smaller_with_leak_zero_on_ce_path() -> None:
    """CE-only path: full leak (1) yields non-zero encoder grad; zero leak yields zero grad on prefix slice."""
    base = {
        "d_model": 128,
        "encoder_layers": 1,
        "nhead": 4,
        "dim_feedforward": 256,
        "dropout": 0.0,
        "use_cls": True,
        "predictor_prefix_dims": 64,
        "jepa_square_embed_dim": 16,
        "predictor_hidden": 128,
        "predictor_depth": 2,
        "from_to_head_hidden": 64,
        "from_to_head_depth": 2,
        "probe_board_recon_hidden": 32,
        "probe_meta_hidden": 32,
    }
    z = torch.randn(1, 128, requires_grad=True)

    m0 = ChessJEPAV4Builder.build({**base, "move_head_prefix_leak": 0.0})
    m0.train()
    logits0 = m0.forward_from_logits(z, move_head_prefix_leak=0.0)
    loss0 = logits0.sum()
    loss0.backward()
    g0 = z.grad.clone()

    z1 = torch.randn(1, 128, requires_grad=True)
    m1 = ChessJEPAV4Builder.build({**base, "move_head_prefix_leak": 1.0})
    m1.train()
    logits1 = m1.forward_from_logits(z1, move_head_prefix_leak=1.0)
    loss1 = logits1.sum()
    loss1.backward()
    g1 = z1.grad.clone()

    assert g0[:, :64].abs().max().item() == 0.0
    assert g1[:, :64].abs().max().item() > 0.0
