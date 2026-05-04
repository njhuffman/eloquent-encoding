"""Tests for moved/placed category gathering."""

from __future__ import annotations

import torch

from jepa2.config import BOARD_CHANNELS

from jepa3.board_square_categories import square_categories_from_board_tensor

from world_model.action_tokens import moved_placed_categories_from_move


def test_moved_placed_gather_matches_manual_index() -> None:
    torch.manual_seed(0)
    b = 4
    pre = torch.randn(b, 8, 8, BOARD_CHANNELS)
    post = torch.randn(b, 8, 8, BOARD_CHANNELS)
    fs = torch.tensor([0, 15, 33, 63], dtype=torch.long)
    ts = torch.tensor([8, 20, 40, 62], dtype=torch.long)
    moved, placed = moved_placed_categories_from_move(pre, post, fs, ts)
    cats_pre = square_categories_from_board_tensor(pre)
    cats_post = square_categories_from_board_tensor(post)
    bi = torch.arange(b, dtype=torch.long)
    assert torch.equal(moved, cats_pre[bi, fs])
    assert torch.equal(placed, cats_post[bi, ts])


def test_moved_placed_raises_on_batch_mismatch() -> None:
    pre = torch.randn(2, 8, 8, BOARD_CHANNELS)
    post = torch.randn(2, 8, 8, BOARD_CHANNELS)
    fs = torch.zeros(3, dtype=torch.long)
    ts = torch.zeros(3, dtype=torch.long)
    try:
        moved_placed_categories_from_move(pre, post, fs, ts)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
