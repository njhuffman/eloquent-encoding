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


_WHITE_WDL = {"1-0": 2, "1/2-1/2": 1, "0-1": 0}  # White's win/draw/loss


def collect_candidate_positions(
    game: chess.pgn.Game,
    *,
    skip_opening_plies: int,
    exclude_single_legal_move: bool,
) -> tuple[list[chess.Move], list[tuple[int, int, int, int, int, chess.Move]]]:
    """Mainline-order candidates before each half-move.

    Returns ``mainline`` and rows ``(ply, side_to_move 0/1, elo_to_move, opp_elo, result, played_move)``.
    ``result`` is loss=0/draw=1/win=2 from the side-to-move's perspective. Games whose
    Result header is not a terminated outcome (e.g. ``*``) are dropped (returns ``[], []``).
    """
    white_elo_h = _parse_elo(game.headers, "white")
    black_elo_h = _parse_elo(game.headers, "black")
    if white_elo_h is None or black_elo_h is None:
        return [], []
    white_wdl = _WHITE_WDL.get(game.headers.get("Result", "*"))
    if white_wdl is None:
        return [], []

    mainline = list(game.mainline_moves())
    board = game.board()
    ply = 0
    out: list[tuple[int, int, int, int, int, chess.Move]] = []
    for move in mainline:
        if ply >= skip_opening_plies:
            if exclude_single_legal_move and board.legal_moves.count() < 2:
                board.push(move)
                ply += 1
                continue
            if board.turn == chess.WHITE:
                stm, elo_tm, opp_elo, result = 0, white_elo_h, black_elo_h, white_wdl
            else:
                stm, elo_tm, opp_elo, result = 1, black_elo_h, white_elo_h, 2 - white_wdl
            out.append((ply, stm, elo_tm, opp_elo, result, move))
        board.push(move)
        ply += 1
    return mainline, out
