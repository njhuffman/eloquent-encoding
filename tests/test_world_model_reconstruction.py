"""Reconstruction CE heads (patch categories, turn from CLS, legal-origin per square)."""

from __future__ import annotations

import torch

from jepa2.config import BOARD_CHANNELS

from jepa3.board_square_categories import NUM_SQUARE_CATEGORIES, square_categories_from_board_tensor

from world_model.architectures.chess_world_model_v1 import ChessWorldModelV1, resolve_architecture_config
from world_model.loss import world_model_loss_forward


def test_world_model_reconstruction_loss_finite() -> None:
    torch.manual_seed(2)
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
            "recon_piece_head_hidden": 64,
            "recon_turn_head_hidden": 64,
            "recon_can_move_head_hidden": 64,
        }
    )
    m = ChessWorldModelV1(cfg)
    m.train()
    b = 2
    board = torch.randn(b, 8, 8, BOARD_CHANNELS)
    board_post = torch.randn(b, 8, 8, BOARD_CHANNELS)
    fs = torch.randint(0, 64, (b,), dtype=torch.long)
    ts = torch.randint(0, 64, (b,), dtype=torch.long)
    z_g, patch_on, patch_hat = m.encode_online_with_jepa_and_patches(board, board_post, fs, ts)
    _z_pos, patch_pos = m.encode_target_with_tokens(board_post)
    recon = m.forward_reconstruction_logits(z_g, patch_on)
    assert recon["piece_logits"].shape == (b, 64, NUM_SQUARE_CATEGORIES)
    assert recon["turn_logits"].shape == (b, 2)
    assert recon["can_move_logits"].shape == (b, 64, 2)

    labels_sq = square_categories_from_board_tensor(board)
    turn_y = (board[:, 0, 0, 12] > 0.5).long()

    fl = m.forward_from_logits(z_g, patch_on, None)
    fm = torch.zeros(b, 64)
    for i in range(b):
        fm[i, fs[i].item()] = 1.0

    loss, metrics = world_model_loss_forward(
        z_g,
        patch_on,
        patch_hat,
        patch_pos,
        fl,
        fs,
        fm,
        jepa_patch_weight=0.5,
        from_sq_ce_weight=0.0,
        sq_ce_label_smoothing=0.0,
        vicreg={"inv_coef": 0.05, "var_coef": 0.1, "cov_coef": 0.0, "std_target": 1.0},
        use_amp_cuda=False,
        piece_recon_logits=recon["piece_logits"],
        piece_recon_labels=labels_sq,
        turn_recon_logits=recon["turn_logits"],
        turn_recon_labels=turn_y,
        recon_piece_ce_weight=0.2,
        recon_turn_ce_weight=0.1,
    )
    assert torch.isfinite(loss)
    assert "recon_piece_ce" in metrics
    assert "recon_turn_ce" in metrics


def test_world_model_can_move_recon_loss_finite() -> None:
    torch.manual_seed(3)
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
            "recon_can_move_head_hidden": 64,
        }
    )
    m = ChessWorldModelV1(cfg)
    m.train()
    b = 2
    board = torch.randn(b, 8, 8, BOARD_CHANNELS)
    board_post = torch.randn(b, 8, 8, BOARD_CHANNELS)
    fs = torch.randint(0, 64, (b,), dtype=torch.long)
    ts = torch.randint(0, 64, (b,), dtype=torch.long)
    z_g, patch_on, patch_hat = m.encode_online_with_jepa_and_patches(board, board_post, fs, ts)
    _z_pos, patch_pos = m.encode_target_with_tokens(board_post)
    recon = m.forward_reconstruction_logits(z_g, patch_on)
    assert recon["can_move_logits"].shape == (b, 64, 2)

    fl = m.forward_from_logits(z_g, patch_on, None)
    fm = torch.zeros(b, 64)
    for i in range(b):
        fm[i, fs[i].item()] = 1.0
    can_labels = (torch.rand(b, 64, device=board.device) > 0.6).long()

    loss, metrics = world_model_loss_forward(
        z_g,
        patch_on,
        patch_hat,
        patch_pos,
        fl,
        fs,
        fm,
        jepa_patch_weight=0.5,
        from_sq_ce_weight=0.0,
        sq_ce_label_smoothing=0.0,
        vicreg={"inv_coef": 0.05, "var_coef": 0.1, "cov_coef": 0.0, "std_target": 1.0},
        use_amp_cuda=False,
        recon_piece_ce_weight=0.0,
        recon_turn_ce_weight=0.0,
        can_move_logits=recon["can_move_logits"],
        can_move_labels=can_labels,
        recon_can_move_ce_weight=0.2,
    )
    assert torch.isfinite(loss)
    assert "recon_can_move_ce" in metrics
