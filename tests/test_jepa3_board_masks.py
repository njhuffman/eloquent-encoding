"""Tests for jepa3 legal square masks."""

from __future__ import annotations

import chess

from jepa3.board_masks import legal_from_square_mask, legal_to_square_mask


def test_startpos_from_mask_includes_e2_and_excludes_e4() -> None:
    board = chess.Board()
    m = legal_from_square_mask(board)
    assert m.shape == (64,)
    e2 = chess.parse_square("e2")
    e4 = chess.parse_square("e4")
    assert m[e2] == 1.0
    assert m[e4] == 0.0


def test_startpos_to_mask_from_e2_includes_e3_e4() -> None:
    board = chess.Board()
    e2 = chess.parse_square("e2")
    m = legal_to_square_mask(board, e2)
    e3 = chess.parse_square("e3")
    e4 = chess.parse_square("e4")
    assert m[e3] == 1.0 or m[e4] == 1.0
    assert float(m.sum()) >= 2.0


def test_true_move_in_support() -> None:
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    assert move in board.legal_moves
    fm = legal_from_square_mask(board)
    tm = legal_to_square_mask(board, move.from_square)
    assert fm[move.from_square] == 1.0
    assert tm[move.to_square] == 1.0


def test_empty_square_not_a_from_square() -> None:
    board = chess.Board()
    m = legal_from_square_mask(board)
    e4 = chess.parse_square("e4")
    assert m[e4] == 0.0
