"""Decode move-sample HDF5 rows into tensors for JEPA / mining."""

from __future__ import annotations

import sys
from pathlib import Path

import chess
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from embedding.board_encoding import board_to_tensor
from move_predictor.encoding import move_from_slots


def row_to_board_and_move(
    fen: str,
    from_sq: int,
    to_sq: int,
    promotion: int,
) -> tuple[chess.Board, chess.Move] | None:
    """
    Return (board_before, played_move) or None if FEN/move invalid or move illegal.
    """
    try:
        board = chess.Board(fen)
    except ValueError:
        return None
    try:
        move = move_from_slots(int(from_sq), int(to_sq), int(promotion))
    except Exception:
        return None
    if move not in board.legal_moves:
        return None
    return board, move


def tensors_for_row(
    fen: str,
    from_sq: int,
    to_sq: int,
    promotion: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """(board_t, board_after_true) float32 (8,8,C) or None."""
    parsed = row_to_board_and_move(fen, from_sq, to_sq, promotion)
    if parsed is None:
        return None
    board, move = parsed
    bt = board_to_tensor(board)
    b2 = board.copy()
    b2.push(move)
    pos = board_to_tensor(b2)
    return bt.astype(np.float32, copy=False), pos.astype(np.float32, copy=False)


def tensor_after_move(board: chess.Board, move: chess.Move) -> np.ndarray:
    b2 = board.copy()
    b2.push(move)
    return board_to_tensor(b2).astype(np.float32, copy=False)
