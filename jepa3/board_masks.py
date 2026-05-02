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


def legal_from_and_to_u64(board: chess.Board, from_sq: int) -> tuple[np.uint64, np.uint64]:
    """Single pass over ``board.legal_moves``: packed legality bitboards for jepa3."""
    from_bits = 0
    to_bits = 0
    fs = int(from_sq)
    for mv in board.legal_moves:
        f = int(mv.from_square)
        from_bits |= 1 << f
        if f == fs:
            to_bits |= 1 << int(mv.to_square)
    return np.uint64(from_bits), np.uint64(to_bits)


def board_from_tensor_full(t: np.ndarray) -> chess.Board:
    """
    Reconstruct ``chess.Board`` from an (8, 8, 18) float32 tensor (``board_to_tensor`` layout).

    Includes castling and en-passant planes so ``legal_moves`` matches packed HDF5 generation.
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
    rights = 0
    if np.max(t[:, :, 13]) > 0.5:
        rights |= chess.BB_H1
    if np.max(t[:, :, 14]) > 0.5:
        rights |= chess.BB_A1
    if np.max(t[:, :, 15]) > 0.5:
        rights |= chess.BB_H8
    if np.max(t[:, :, 16]) > 0.5:
        rights |= chess.BB_A8
    board.castling_rights = rights
    board.ep_square = None
    ep_plane = t[:, :, 17]
    if float(np.max(ep_plane)) > 0.5:
        r, c = np.unravel_index(int(np.argmax(ep_plane)), (8, 8))
        board.ep_square = chess.square(int(c), int(r))
    return board
