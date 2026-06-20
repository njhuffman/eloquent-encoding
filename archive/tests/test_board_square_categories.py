"""Tests for jepa3.board_square_categories mapping."""

from __future__ import annotations

import chess
import pytest
import torch

from embedding.board_encoding import board_to_tensor
from jepa3.board_square_categories import (
    CAT_BK_WITH_CASTLE,
    CAT_BR_NO_CASTLE,
    CAT_BR_WITH_CASTLE,
    CAT_EMPTY_EP_TARGET,
    CAT_WK_NO_CASTLE,
    CAT_WK_WITH_CASTLE,
    CAT_WR_NO_CASTLE,
    CAT_WR_WITH_CASTLE,
    SQ_A8,
    SQ_H8,
    square_categories_from_board_tensor,
)


def _cats(board: chess.Board) -> torch.Tensor:
    t = torch.from_numpy(board_to_tensor(board)).float().unsqueeze(0)
    return square_categories_from_board_tensor(t)[0].cpu()


def test_starting_position_white_rooks_and_king_eligible() -> None:
    b = chess.Board()
    c = _cats(b)
    assert int(c[0]) == CAT_WR_WITH_CASTLE
    assert int(c[7]) == CAT_WR_WITH_CASTLE
    assert int(c[chess.E1]) == CAT_WK_WITH_CASTLE
    assert int(c[chess.E8]) == CAT_BK_WITH_CASTLE


def test_en_passant_empty_square_category() -> None:
    board = chess.Board()
    for uci in ["e2e4", "e7e6", "e4e5", "d7d5"]:
        board.push(chess.Move.from_uci(uci))
    assert board.ep_square == chess.D6
    c = _cats(board)
    assert int(c[int(board.ep_square)]) == CAT_EMPTY_EP_TARGET


def test_white_king_moved_rooks_ineligible() -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    board.push_uci("e1e2")
    c = _cats(board)
    assert int(c[chess.E2]) == CAT_WK_NO_CASTLE
    assert int(c[chess.A1]) == CAT_WR_NO_CASTLE
    assert int(c[chess.H1]) == CAT_WR_NO_CASTLE


def test_black_rooks_corner_eligibility_start() -> None:
    b = chess.Board()
    c = _cats(b)
    assert int(c[SQ_A8]) == CAT_BR_WITH_CASTLE
    assert int(c[SQ_H8]) == CAT_BR_WITH_CASTLE


def test_square_categories_bad_shape() -> None:
    with pytest.raises(ValueError):
        square_categories_from_board_tensor(torch.zeros(2, 8, 8))


def test_batch_runs_on_cuda_if_available() -> None:
    t = torch.from_numpy(board_to_tensor(chess.Board())).float().unsqueeze(0).repeat(3, 1, 1, 1)
    if torch.cuda.is_available():
        out = square_categories_from_board_tensor(t.cuda())
        assert out.shape == (3, 64)
        assert out.device.type == "cuda"
    else:
        out = square_categories_from_board_tensor(t)
        assert out.shape == (3, 64)


def test_num_categories_is_18() -> None:
    from jepa3.board_square_categories import NUM_SQUARE_CATEGORIES

    assert NUM_SQUARE_CATEGORIES == 18
