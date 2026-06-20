"""Chess move ↔ square indices and promotion codes stored in HDF5."""

from __future__ import annotations

import chess

# Stored in HDF5 promotion column: no promotion
PROMOTION_NONE = 0


def move_to_from_to(move: chess.Move) -> tuple[int, int]:
    return int(move.from_square), int(move.to_square)


def promotion_code(move: chess.Move) -> int:
    """0 if not a promotion; else python-chess piece_type of promoted piece (2..5)."""
    if move.promotion is None:
        return PROMOTION_NONE
    return int(move.promotion)


def code_to_promotion_piece(code: int) -> chess.PieceType | None:
    if code == PROMOTION_NONE:
        return None
    return chess.PieceType(code)


def move_from_slots(
    from_sq: int,
    to_sq: int,
    promotion_code_val: int,
) -> chess.Move:
    prom = code_to_promotion_piece(promotion_code_val)
    return chess.Move(from_sq, to_sq, promotion=prom)
