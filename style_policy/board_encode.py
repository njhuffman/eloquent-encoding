"""Live-play helpers: encode a python-chess Board into the model's packed format
(inverse of packed_codec) and derive legal-move bitboards. Used by bots at inference."""
from __future__ import annotations
import numpy as np
import chess
from style_policy.packed_codec import PACKED_BOARD_LEN


def board_to_packed(board: chess.Board) -> np.ndarray:
    """chess.Board -> uint8 (PACKED_BOARD_LEN,) matching the on-disk j3 packed layout."""
    out = np.zeros(PACKED_BOARD_LEN, dtype=np.uint8)
    for sq, piece in board.piece_map().items():
        nib = piece.piece_type + (0 if piece.color == chess.WHITE else 6)  # 1-6 white P..K, 7-12 black
        if sq % 2 == 0:
            out[sq // 2] |= nib & 0x0F
        else:
            out[sq // 2] |= (nib & 0x0F) << 4
    meta = 0
    if board.turn == chess.WHITE:
        meta |= 1
    if board.castling_rights & chess.BB_H1:
        meta |= 2
    if board.castling_rights & chess.BB_A1:
        meta |= 4
    if board.castling_rights & chess.BB_H8:
        meta |= 8
    if board.castling_rights & chess.BB_A8:
        meta |= 16
    out[32] = meta
    out[33] = board.ep_square if board.ep_square is not None else 255
    return out


def legal_from_u64(board: chess.Board) -> int:
    bb = 0
    for mv in board.legal_moves:
        bb |= 1 << mv.from_square
    return bb


def legal_to_u64(board: chess.Board, from_sq: int) -> int:
    bb = 0
    for mv in board.legal_moves:
        if mv.from_square == from_sq:
            bb |= 1 << mv.to_square
    return bb
