"""Chess helpers: legal moves, candidate sampling, successor tensors (uses jepa move_row_codec)."""

from __future__ import annotations

import random
from typing import Any

import chess
import numpy as np
import torch

from jepa.move_row_codec import row_to_board_and_move, tensor_after_move


def parse_row(
    fen: str,
    from_sq: int,
    to_sq: int,
    promotion: int,
) -> tuple[chess.Board, chess.Move, list[chess.Move]] | None:
    parsed = row_to_board_and_move(str(fen), int(from_sq), int(to_sq), int(promotion))
    if parsed is None:
        return None
    board, played = parsed
    legals = list(board.legal_moves)
    return board, played, legals


def pick_candidate_moves(
    legals: list[chess.Move],
    played: chess.Move,
    M_cap: int | None,
    rng: random.Random,
) -> tuple[list[chess.Move], int, int]:
    """
    If len(legals) <= M_cap or M_cap is None, return all legals and label index of played.
    Else return [played] + (M_cap-1) uniform random wrong moves; label index is 0.
    Returns (moves, label_index, n_legals_full).
    """
    n = len(legals)
    if M_cap is None or n <= M_cap:
        moves = list(legals)
        return moves, moves.index(played), n
    wrong = [m for m in legals if m != played]
    need = min(M_cap - 1, len(wrong))
    sampled = rng.sample(wrong, need)
    moves = [played] + sampled
    return moves, 0, n


def successor_stack(board: chess.Board, moves: list[chess.Move]) -> np.ndarray:
    """(K, H, W, C) float32."""
    arrs = [tensor_after_move(board, m) for m in moves]
    return np.stack(arrs, axis=0).astype(np.float32, copy=False)


def prepare_batch_tensors(
    rows: list[dict[str, Any]],
    M_cap: int | None,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """
    ``rows``: each dict has fen, elo_to_move, from_sq, to_sq, promotion, board_t (H,W,C) float32.

    Returns:
      board_t (B,8,8,C), elo (B,), succ (B,M,8,8,C), mask (B,M), labels (B,), n_full (B,) as long
    """
    B = len(rows)
    boards: list[np.ndarray] = []
    elos: list[float] = []
    succ_blocks: list[np.ndarray] = []
    masks: list[list[float]] = []
    labels: list[int] = []
    n_full: list[int] = []

    for r in rows:
        boards.append(np.asarray(r["board_t"], dtype=np.float32))
        elos.append(float(r["elo_to_move"]))
        parsed = parse_row(str(r["fen"]), int(r["from_sq"]), int(r["to_sq"]), int(r["promotion"]))
        if parsed is None:
            raise ValueError(f"invalid row fen/move: {r.get('fen')}")
        board, played, legals = parsed
        moves, lab, n = pick_candidate_moves(legals, played, M_cap, rng)
        succ_blocks.append(successor_stack(board, moves))
        masks.append([1.0] * len(moves))
        labels.append(lab)
        n_full.append(n)

    M = max(s.shape[0] for s in succ_blocks)
    h, w, c = succ_blocks[0].shape[1:]
    succ = np.zeros((B, M, h, w, c), dtype=np.float32)
    mask = np.zeros((B, M), dtype=np.float32)
    for i, s in enumerate(succ_blocks):
        k = s.shape[0]
        succ[i, :k] = s
        mask[i, :k] = 1.0

    board_t = torch.from_numpy(np.stack(boards, axis=0))
    elo_t = torch.tensor(elos, dtype=torch.float32)
    succ_t = torch.from_numpy(succ)
    mask_t = torch.from_numpy(mask)
    labels_t = torch.tensor(labels, dtype=torch.long)
    n_full_t = torch.tensor(n_full, dtype=torch.long)
    return board_t, elo_t, succ_t, mask_t, labels_t, n_full_t
