"""Legal from/to square masks for 64-way CE (python-chess square indices 0..63)."""

from __future__ import annotations

import numpy as np
import chess


def legal_from_square_mask(board: chess.Board) -> np.ndarray:
    """Shape (64,) float32: 1.0 iff at least one legal move originates from that square."""
    m = np.zeros(64, dtype=np.float32)
    for mv in board.legal_moves:
        m[int(mv.from_square)] = 1.0
    return m


def legal_to_square_mask(board: chess.Board, from_sq: int) -> np.ndarray:
    """Shape (64,) float32: 1.0 iff square is a legal destination from ``from_sq``."""
    m = np.zeros(64, dtype=np.float32)
    fs = int(from_sq)
    for mv in board.legal_moves:
        if int(mv.from_square) == fs:
            m[int(mv.to_square)] = 1.0
    return m
