"""Chess move -> square indices and promotion code for HDF5 storage."""
from __future__ import annotations
import chess

PROMOTION_NONE = 0


def move_to_from_to(move: chess.Move) -> tuple[int, int]:
    return int(move.from_square), int(move.to_square)


def promotion_code(move: chess.Move) -> int:
    """0 if not a promotion; else python-chess piece_type of the promoted piece (2..5)."""
    return PROMOTION_NONE if move.promotion is None else int(move.promotion)
