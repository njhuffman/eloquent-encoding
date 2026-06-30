"""Shared candidate extraction for PGN games (dataset_generation + jepa3 packed build)."""

from __future__ import annotations

import collections
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
) -> tuple[list[chess.Move], list[tuple[int, int, int, int, int, chess.Move, list[tuple[int, int, int]]]]]:
    """Mainline-order candidates before each half-move.

    Returns ``mainline`` and rows ``(ply, side_to_move 0/1, elo_to_move, opp_elo, result, played_move, hist)``.
    ``result`` is loss=0/draw=1/win=2 from the side-to-move's perspective. Games whose
    Result header is not a terminated outcome (e.g. ``*``) are dropped (returns ``[], []``).

    ``hist`` is a length-4 list of ``(from_sq, to_sq, cap)`` tuples, newest-first (hist[0] is
    the immediately preceding ply). Absent plies are padded with ``(-1, -1, 0)``.
    ``cap`` is 0 for no capture, else the python-chess piece_type (PAWN=1..QUEEN=5); en passant -> 1.
    """
    white_elo_h = _parse_elo(game.headers, "white")
    black_elo_h = _parse_elo(game.headers, "black")
    if white_elo_h is None or black_elo_h is None:
        return [], []
    white_wdl = _WHITE_WDL.get(game.headers.get("Result", "*"))
    if white_wdl is None:
        return [], []

    _ABSENT = (-1, -1, 0)

    mainline = list(game.mainline_moves())
    board = game.board()
    ply = 0
    # Rolling deque of (from_sq, to_sq, cap) for each pushed move; most-recent at right.
    recent: collections.deque[tuple[int, int, int]] = collections.deque(maxlen=4)
    out: list[tuple] = []
    for move in mainline:
        # Compute capture type BEFORE pushing the move.
        if not board.is_capture(move):
            cap = 0
        elif board.is_en_passant(move):
            cap = chess.PAWN  # == 1
        else:
            piece = board.piece_at(move.to_square)
            cap = piece.piece_type if piece is not None else 0

        if ply >= skip_opening_plies:
            if exclude_single_legal_move and board.legal_moves.count() < 2:
                board.push(move)
                recent.append((move.from_square, move.to_square, cap))
                ply += 1
                continue
            if board.turn == chess.WHITE:
                stm, elo_tm, opp_elo, result = 0, white_elo_h, black_elo_h, white_wdl
            else:
                stm, elo_tm, opp_elo, result = 1, black_elo_h, white_elo_h, 2 - white_wdl
            # Build hist newest-first from the deque (which is oldest-at-left).
            hist: list[tuple[int, int, int]] = list(reversed(recent))
            # Pad to length 4 with absent sentinel.
            while len(hist) < 4:
                hist.append(_ABSENT)
            out.append((ply, stm, elo_tm, opp_elo, result, move, hist))
        board.push(move)
        recent.append((move.from_square, move.to_square, cap))
        ply += 1
    return mainline, out
