"""Forward smoke test for chess_world_model_v1."""

from __future__ import annotations

import torch

from jepa2.config import BOARD_CHANNELS

from jepa3.board_square_categories import NUM_SQUARE_CATEGORIES

from world_model.architectures.chess_world_model_v1 import ChessWorldModelV1, resolve_architecture_config


def test_chess_world_model_v1_encode_shapes() -> None:
    torch.manual_seed(0)
    cfg = resolve_architecture_config(
        {
            "d_model": 64,
            "encoder_layers": 1,
            "nhead": 4,
            "dim_feedforward": 128,
            "predictor_encoder_layers": 1,
            "predictor_nhead": 4,
            "predictor_dim_feedforward": 128,
            "predictor_dropout": 0.0,
            "from_pointer_dim": 64,
            "from_pointer_elo_dim": 0,
        }
    )
    m = ChessWorldModelV1(cfg)
    m.eval()
    b = 2
    board = torch.randn(b, 8, 8, BOARD_CHANNELS)
    board_post = torch.randn(b, 8, 8, BOARD_CHANNELS)
    fs = torch.randint(0, 64, (b,), dtype=torch.long)
    ts = torch.randint(0, 64, (b,), dtype=torch.long)
    z_g, patch_on, patch_hat = m.encode_online_with_jepa_and_patches(board, board_post, fs, ts)
    assert z_g.shape == (b, 64)
    assert patch_on.shape == (b, 64, 64)
    assert patch_hat.shape == patch_on.shape

    z_t, patch_t = m.encode_target_with_tokens(board)
    assert z_t.shape == (b, 64)
    assert patch_t.shape == (b, 64, 64)

    fl = m.forward_from_logits(z_g, patch_on, None)
    assert fl.shape == (b, 64)

    recon = m.forward_reconstruction_logits(z_g, patch_on)
    assert recon["piece_logits"].shape == (b, 64, NUM_SQUARE_CATEGORIES)
    assert recon["turn_logits"].shape == (b, 2)
    assert recon["can_move_logits"].shape == (b, 64, 2)
