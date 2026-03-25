"""
Derive probe targets from 8×8×18 board tensors (same layout as embedding.board_encoding).
JEPA HDF5 has no `meta` dataset; labels are computed from `board_t` (+ `elo` column).
"""

from __future__ import annotations

import numpy as np
import chess


def board_from_tensor(t: np.ndarray) -> chess.Board:
    """
    Reconstruct piece positions and side to move for geometry (e.g. is_check).
    Castling / EP flags are omitted; sufficient for python-chess attack / in_check.
    """
    t = np.asarray(t, dtype=np.float32)
    board = chess.Board.empty()
    for row in range(8):
        for col in range(8):
            sq = chess.square(col, row)
            w = t[row, col, 0:6]
            b = t[row, col, 6:12]
            if w.max() > 0.5:
                pt = int(np.argmax(w)) + 1
                board.set_piece_at(sq, chess.Piece(pt, chess.WHITE))
            elif b.max() > 0.5:
                pt = int(np.argmax(b)) + 1
                board.set_piece_at(sq, chess.Piece(pt, chess.BLACK))
    board.turn = chess.WHITE if t[0, 0, 12] > 0.5 else chess.BLACK
    return board


def piece_counts_from_board_tensor(t: np.ndarray) -> tuple[float, float]:
    """Return (white_piece_count, black_piece_count) for one board (8,8,18)."""
    t = np.asarray(t, dtype=np.float32)
    n_white = float(t[:, :, 0:6].sum())
    n_black = float(t[:, :, 6:12].sum())
    return n_white, n_black


def piece_counts_batch(boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """boards (N, 8, 8, 18) -> (n_white, n_black) each (N,)."""
    b = np.asarray(boards, dtype=np.float32)
    w = b[:, :, :, 0:6].sum(axis=(1, 2, 3))
    bl = b[:, :, :, 6:12].sum(axis=(1, 2, 3))
    return w.astype(np.float32), bl.astype(np.float32)


def in_check_batch(boards: np.ndarray) -> np.ndarray:
    """(N, 8, 8, 18) -> (N,) float32 0/1."""
    out = np.zeros(boards.shape[0], dtype=np.float32)
    for i in range(boards.shape[0]):
        try:
            out[i] = 1.0 if board_from_tensor(boards[i]).is_check() else 0.0
        except (ValueError, AssertionError):
            out[i] = 0.0
    return out
