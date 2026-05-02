"""Shared candidate extraction for PGN games (dataset_generation + jepa3 packed build)."""

from __future__ import annotations

import chess
import chess.pgn


def _parse_elo(headers: chess.pgn.Headers, color: str) -> int | None:
    key = f"{color.capitalize()}Elo"
    raw = headers.get(key)
    if raw is None or raw == "?":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def board_at_ply(mainline: list[chess.Move], ply: int) -> chess.Board:
    """Position before mainline[ply] (same indexing as collect_candidate_positions)."""
    b = chess.Board()
    for i in range(ply):
        b.push(mainline[i])
    return b


def collect_candidate_positions(
    game: chess.pgn.Game,
    *,
    skip_opening_plies: int,
    exclude_single_legal_move: bool,
) -> tuple[list[chess.Move], list[tuple[int, int, int, chess.Move]]]:
    """
    Mainline-order candidates before each half-move.

    Returns ``mainline`` and rows ``(ply, side_to_move 0/1, elo_to_move, played_move)``.
    ``ply`` is the half-move index (0 before first move); replay with ``board_at_ply``.
    """
    white_elo_h = _parse_elo(game.headers, "white")
    black_elo_h = _parse_elo(game.headers, "black")
    if white_elo_h is None or black_elo_h is None:
        return [], []

    mainline = list(game.mainline_moves())
    board = game.board()
    ply = 0
    out: list[tuple[int, int, int, chess.Move]] = []
    for move in mainline:
        if ply >= skip_opening_plies:
            if exclude_single_legal_move:
                if board.legal_moves.count() < 2:
                    board.push(move)
                    ply += 1
                    continue
            stm = 0 if board.turn == chess.WHITE else 1
            elo_tm = white_elo_h if board.turn == chess.WHITE else black_elo_h
            out.append((ply, stm, elo_tm, move))
        board.push(move)
        ply += 1
    return mainline, out
