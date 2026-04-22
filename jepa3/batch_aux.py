"""CPU-side batch tensors for jepa3 (post-move boards + legality masks)."""

from __future__ import annotations

from typing import Any

import numpy as np

from jepa.move_row_codec import tensor_after_move
from jepa2.chess_io import parse_row
from jepa3.board_masks import legal_from_square_mask, legal_to_square_mask


def post_move_boards_np(rows: list[dict[str, Any]]) -> np.ndarray:
    """(B, 8, 8, C) float32 boards after the played move."""
    blocks: list[np.ndarray] = []
    for r in rows:
        parsed = parse_row(str(r["fen"]), int(r["from_sq"]), int(r["to_sq"]), int(r["promotion"]))
        if parsed is None:
            raise ValueError(f"invalid row fen/move: {r.get('fen')}")
        board, played, _ = parsed
        blocks.append(tensor_after_move(board, played))
    return np.stack(blocks, axis=0).astype(np.float32, copy=False)


def legal_masks_np(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """from_mask (B,64), to_mask (B,64) float32."""
    b = len(rows)
    fm = np.zeros((b, 64), dtype=np.float32)
    tm = np.zeros((b, 64), dtype=np.float32)
    for i, r in enumerate(rows):
        parsed = parse_row(str(r["fen"]), int(r["from_sq"]), int(r["to_sq"]), int(r["promotion"]))
        if parsed is None:
            raise ValueError(f"invalid row fen/move: {r.get('fen')}")
        board, _, _ = parsed
        fm[i] = legal_from_square_mask(board)
        tm[i] = legal_to_square_mask(board, int(r["from_sq"]))
    return fm, tm
