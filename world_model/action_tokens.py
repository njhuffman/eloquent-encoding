"""Action-side labels for patch JEPA predictor (aligned with BoardEncoderV3 categories)."""

from __future__ import annotations

import torch

from jepa3.board_square_categories import square_categories_from_board_tensor


def moved_placed_categories_from_move(
    board_pre: torch.Tensor,
    board_post: torch.Tensor,
    from_sq: torch.Tensor,
    to_sq: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Moved / placed square categories for the four-token action prefix.

    ``moved_cat[b]`` is ``square_categories_from_board_tensor(board_pre)[b, from_sq[b]]``
    (piece on the origin square before the move). ``placed_cat[b]`` is the category on
    ``to_sq`` after the move (e.g. promoted queen). Uses the same 18-way vocabulary as
    ``BoardEncoderV3.piece_emb``.

    Dataset rows use legal moves; castling is stored as king ``from_sq`` / ``to_sq``
    (see ``move_predictor.encoding.move_to_from_to``). Categories include king/rook
    castle-split ids consistent with the encoder.
    """
    if board_pre.shape != board_post.shape:
        raise ValueError(f"board_pre and board_post shapes must match; got {board_pre.shape} vs {board_post.shape}")
    b = board_pre.shape[0]
    if from_sq.shape != (b,) or to_sq.shape != (b,):
        raise ValueError(f"from_sq and to_sq must be (B,) with B={b}; got {tuple(from_sq.shape)}, {tuple(to_sq.shape)}")
    cats_pre = square_categories_from_board_tensor(board_pre)
    cats_post = square_categories_from_board_tensor(board_post)
    fs = from_sq.long().clamp(0, 63)
    ts = to_sq.long().clamp(0, 63)
    bi = torch.arange(b, device=board_pre.device, dtype=torch.long)
    moved = cats_pre[bi, fs]
    placed = cats_post[bi, ts]
    return moved, placed
