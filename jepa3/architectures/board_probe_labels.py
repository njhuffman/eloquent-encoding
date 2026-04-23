"""Targets for chess_jepa_v4 board reconstruction and meta aux tasks from (B, 8, 8, 18) tensors."""

from __future__ import annotations

import torch

from jepa2.config import BOARD_HEIGHT, BOARD_WIDTH


def piece_labels_64_from_board(board: torch.Tensor) -> torch.Tensor:
    """
    Per-square 13-class labels: 0..11 = piece plane one-hot, 12 = empty.

    ``board`` (B, H, W, C) with piece occupancy in channels 0..11 (same layout as
    ``embedding.board_encoding``). Flatten order matches ``board.reshape(B, H*W, C)``.
    """
    if board.ndim != 4:
        raise ValueError(f"board must be (B,H,W,C), got {tuple(board.shape)}")
    b, h, w, c = board.shape
    if h != BOARD_HEIGHT or w != BOARD_WIDTH or c < 12:
        raise ValueError(f"expected H=W=8 and C>=12, got {tuple(board.shape)}")
    pieces = board[:, :, :, :12].float()
    # (B, H, W)
    occ = pieces.sum(dim=-1)
    has_piece = occ > 0.5
    argm = pieces.argmax(dim=-1).clamp(0, 11)
    empty = torch.full_like(argm, 12, dtype=torch.long)
    labels_hw = torch.where(has_piece, argm, empty)
    return labels_hw.reshape(b, h * w).long()


def meta_targets_from_board(board: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Targets derived from planes 12..17 (turn, castling layers, en passant).

    Returns tensors on ``board.device``, dtype float for BCE targets and long for CE.
    """
    if board.ndim != 4 or board.shape[-1] < 18:
        raise ValueError(f"board must have at least 18 channels, got {tuple(board.shape)}")
    b = board.shape[0]
    turn = (board[:, 0, 0, 12] > 0.5).float()
    castle = (board[:, 0, 0, 13:17] > 0.5).float()
    ep_plane = board[:, :, :, 17].reshape(b, -1).float()
    ep_max, ep_idx = ep_plane.max(dim=1)
    ep_class = torch.where(ep_max > 0.5, ep_idx + 1, torch.zeros(b, device=board.device, dtype=torch.long))
    return {"turn": turn, "castle": castle, "ep_class": ep_class}
