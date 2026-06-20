"""
Encode a python-chess Board as an 8x8x18 tensor and extract 8x8x12 piece mask for decoder target.
"""

import numpy as np
import chess

from .config import BOARD_HEIGHT, BOARD_WIDTH, BOARD_CHANNELS, PIECE_PLANES

# Piece type to plane index (0-5 white, 6-11 black). python-chess: PAWN=1, KNIGHT=2, BISHOP=3, ROOK=4, QUEEN=5, KING=6
WHITE_PLANES = list(range(0, 6))   # P, N, B, R, Q, K
BLACK_PLANES = list(range(6, 12))


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Encode board state as (8, 8, 18) float32 tensor.
    - Planes 0-5: white P, N, B, R, Q, K (one-hot per square).
    - Planes 6-11: black P, N, B, R, Q, K.
    - Plane 12: turn (1.0 = white to move, 0.0 = black to move).
    - Planes 13-16: castling (white K-side, white Q-side, black K-side, black Q-side), full 8x8 layer 0 or 1.
    - Plane 17: en passant target square (1.0 on that square, 0 elsewhere; 0 everywhere if no ep).
    """
    out = np.zeros((BOARD_HEIGHT, BOARD_WIDTH, BOARD_CHANNELS), dtype=np.float32)

    for sq in chess.SQUARES:
        row = chess.square_rank(sq)
        col = chess.square_file(sq)
        piece = board.piece_at(sq)
        if piece is not None:
            pt = piece.piece_type  # 1..6
            plane = (pt - 1) + (6 if not piece.color else 0)
            out[row, col, plane] = 1.0

    # Plane 12: turn (white = 1, black = 0)
    out[:, :, 12] = 1.0 if board.turn == chess.WHITE else 0.0

    # Planes 13-16: castling (full layer)
    if board.has_kingside_castling_rights(chess.WHITE):
        out[:, :, 13] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        out[:, :, 14] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        out[:, :, 15] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        out[:, :, 16] = 1.0

    # Plane 17: en passant
    if board.ep_square is not None:
        ep_row = chess.square_rank(board.ep_square)
        ep_col = chess.square_file(board.ep_square)
        out[ep_row, ep_col, 17] = 1.0

    return out


def get_piece_mask_8x8x12(board_tensor: np.ndarray) -> np.ndarray:
    """
    Extract the first 12 channels (piece positions only) from an 8x8x18 board tensor.
    Used as decoder target; shape (8, 8, 12).
    """
    return board_tensor[:, :, :PIECE_PLANES].copy()
