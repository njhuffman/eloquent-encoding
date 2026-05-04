"""Map (8,8,18) board tensors to per-square category IDs for embedding lookup.

18 categories: empty, empty+EP target, then per color (white 2–9, black 10–17):
pawn, bishop, knight, queen, king±castling rights, rook±castling rights.

Planes match ``embedding.board_encoding.board_to_tensor``: pieces 0–11, turn 12,
castling 13–16 (H1, A1, H8, A8 rights), EP target 17.

Row-major flatten: ``tensor[row,col]`` → flat ``row * 8 + col`` (python-chess square index).
"""

from __future__ import annotations

import torch

from jepa2.config import BOARD_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH

# --- 18 category ids (single nn.Embedding table) ---
CAT_EMPTY = 0
CAT_EMPTY_EP_TARGET = 1

# White: pawn, bishop, knight, queen, king±castle, rook±castle
CAT_WP = 2
CAT_WB = 3
CAT_WN = 4
CAT_WQ = 5
CAT_WK_WITH_CASTLE = 6
CAT_WK_NO_CASTLE = 7
CAT_WR_WITH_CASTLE = 8
CAT_WR_NO_CASTLE = 9

# Black: same order
CAT_BP = 10
CAT_BB = 11
CAT_BN = 12
CAT_BQ = 13
CAT_BK_WITH_CASTLE = 14
CAT_BK_NO_CASTLE = 15
CAT_BR_WITH_CASTLE = 16
CAT_BR_NO_CASTLE = 17

NUM_SQUARE_CATEGORIES = 18

SQ_A1 = 0
SQ_H1 = 7
SQ_A8 = 56
SQ_H8 = 63


def square_categories_from_board_tensor(board: torch.Tensor) -> torch.Tensor:
    """
    Args:
        board: (B, H, W, C) with C >= 18, float or half/bfloat16.

    Returns:
        Long tensor (B, 64) category index in ``0 .. NUM_SQUARE_CATEGORIES - 1``.
    """
    if board.ndim != 4:
        raise ValueError(f"board must be 4D (B,H,W,C), got shape {tuple(board.shape)}")
    if board.shape[1] != BOARD_HEIGHT or board.shape[2] != BOARD_WIDTH:
        raise ValueError(f"board spatial dims must be ({BOARD_HEIGHT},{BOARD_WIDTH}), got {board.shape[1:3]}")
    if board.shape[-1] < BOARD_CHANNELS:
        raise ValueError(f"board needs at least {BOARD_CHANNELS} channels, got {board.shape[-1]}")

    b = board.shape[0]
    device = board.device
    flat = board.reshape(b, 64, board.shape[-1]).float()
    piece_planes = flat[:, :, :12]
    maxv, midx = piece_planes.max(dim=-1)
    empty = maxv <= 0.5
    ep_here = flat[:, :, 17] > 0.5

    wks = board[:, :, :, 13].amax(dim=(1, 2)) > 0.5
    wqs = board[:, :, :, 14].amax(dim=(1, 2)) > 0.5
    bks = board[:, :, :, 15].amax(dim=(1, 2)) > 0.5
    bqs = board[:, :, :, 16].amax(dim=(1, 2)) > 0.5

    sq = torch.arange(64, device=device, dtype=torch.long).view(1, 64).expand(b, -1)

    occ_cat = torch.zeros(b, 64, dtype=torch.long, device=device)
    pw = ~empty & (midx < 6)
    pb = ~empty & (midx >= 6)

    occ_cat = torch.where(pw & (midx == 0), torch.full_like(occ_cat, CAT_WP), occ_cat)
    occ_cat = torch.where(pw & (midx == 2), torch.full_like(occ_cat, CAT_WB), occ_cat)
    occ_cat = torch.where(pw & (midx == 1), torch.full_like(occ_cat, CAT_WN), occ_cat)
    occ_cat = torch.where(pw & (midx == 4), torch.full_like(occ_cat, CAT_WQ), occ_cat)

    wr_corner_elig = (pw & (midx == 3) & (sq == SQ_A1) & wqs.unsqueeze(1)) | (
        pw & (midx == 3) & (sq == SQ_H1) & wks.unsqueeze(1)
    )
    wr_cat = torch.where(
        wr_corner_elig,
        torch.full_like(occ_cat, CAT_WR_WITH_CASTLE),
        torch.full_like(occ_cat, CAT_WR_NO_CASTLE),
    )
    occ_cat = torch.where(pw & (midx == 3), wr_cat, occ_cat)

    wk_elig = wks | wqs
    wk_pick = torch.where(
        wk_elig.unsqueeze(1),
        torch.full_like(occ_cat, CAT_WK_WITH_CASTLE),
        torch.full_like(occ_cat, CAT_WK_NO_CASTLE),
    )
    occ_cat = torch.where(pw & (midx == 5), wk_pick, occ_cat)

    occ_cat = torch.where(pb & (midx == 6), torch.full_like(occ_cat, CAT_BP), occ_cat)
    occ_cat = torch.where(pb & (midx == 8), torch.full_like(occ_cat, CAT_BB), occ_cat)
    occ_cat = torch.where(pb & (midx == 7), torch.full_like(occ_cat, CAT_BN), occ_cat)
    occ_cat = torch.where(pb & (midx == 10), torch.full_like(occ_cat, CAT_BQ), occ_cat)

    br_corner_elig = (pb & (midx == 9) & (sq == SQ_A8) & bqs.unsqueeze(1)) | (
        pb & (midx == 9) & (sq == SQ_H8) & bks.unsqueeze(1)
    )
    br_cat = torch.where(
        br_corner_elig,
        torch.full_like(occ_cat, CAT_BR_WITH_CASTLE),
        torch.full_like(occ_cat, CAT_BR_NO_CASTLE),
    )
    occ_cat = torch.where(pb & (midx == 9), br_cat, occ_cat)

    bk_elig = bks | bqs
    bk_pick = torch.where(
        bk_elig.unsqueeze(1),
        torch.full_like(occ_cat, CAT_BK_WITH_CASTLE),
        torch.full_like(occ_cat, CAT_BK_NO_CASTLE),
    )
    occ_cat = torch.where(pb & (midx == 11), bk_pick, occ_cat)

    empty_cat = torch.where(
        ep_here,
        torch.full_like(occ_cat, CAT_EMPTY_EP_TARGET),
        torch.full_like(occ_cat, CAT_EMPTY),
    )
    return torch.where(empty, empty_cat, occ_cat)
