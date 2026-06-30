"""Test that collect_candidate_positions emits last-4-ply history tuples correctly."""
from __future__ import annotations

import io
import chess
import chess.pgn

from dataset_generation.candidate_collect import collect_candidate_positions

# Game: 1.e4 d5 2.exd5 c6 3.Nc3 Nf6 4.d4 e6
# Ply 0: e4   (e2->e4, no cap)
# Ply 1: d5   (d7->d5, no cap)
# Ply 2: exd5 (e4->d5, captures pawn; chess.PAWN == 1)
# Ply 3: c6   (c7->c6, no cap)
# Ply 4: Nc3  (b1->c3, no cap)
# Ply 5: Nf6  (g8->f6, no cap)
# Ply 6: d4   (d2->d4, no cap)
# Ply 7: e6   (e7->e6, no cap)
_PGN = """[White "A"]
[Black "B"]
[WhiteElo "1600"]
[BlackElo "1500"]
[Result "1-0"]

1. e4 d5 2. exd5 c6 3. Nc3 Nf6 4. d4 e6 1-0
"""

ABSENT = (-1, -1, 0)


def _parse_game(pgn: str) -> chess.pgn.Game:
    return chess.pgn.read_game(io.StringIO(pgn))


def _rows_by_ply(pgn: str, skip: int = 0):
    game = _parse_game(pgn)
    _, rows = collect_candidate_positions(
        game, skip_opening_plies=skip, exclude_single_legal_move=False
    )
    return {row[0]: row for row in rows}


def test_row_is_7_tuple():
    """Each emitted row must be a 7-tuple (ply, stm, elo_tm, opp_elo, result, move, hist)."""
    by_ply = _rows_by_ply(_PGN)
    row = next(iter(by_ply.values()))
    assert len(row) == 7, f"expected 7-tuple, got {len(row)}"


def test_ply0_has_all_absent_hist():
    """At ply 0, no prior plies exist — all 4 slots must be absent (-1,-1,0)."""
    by_ply = _rows_by_ply(_PGN)
    hist = by_ply[0][6]
    assert len(hist) == 4
    for i, entry in enumerate(hist):
        assert entry == ABSENT, f"hist[{i}] at ply 0 should be absent, got {entry}"


def test_ply1_has_one_entry_rest_absent():
    """At ply 1 (d5 about to be played), only hist[0] = e4 move (no capture)."""
    by_ply = _rows_by_ply(_PGN)
    hist = by_ply[1][6]
    # hist[0] = ply 0: e4 = e2->e4, cap=0
    assert hist[0] == (chess.E2, chess.E4, 0), f"hist[0] at ply 1: {hist[0]}"
    assert hist[1] == ABSENT, f"hist[1] at ply 1 should be absent: {hist[1]}"
    assert hist[2] == ABSENT, f"hist[2] at ply 1 should be absent: {hist[2]}"
    assert hist[3] == ABSENT, f"hist[3] at ply 1 should be absent: {hist[3]}"


def test_ply2_has_two_entries():
    """At ply 2 (exd5 about to be played), hist[0]=d5 move, hist[1]=e4 move."""
    by_ply = _rows_by_ply(_PGN)
    hist = by_ply[2][6]
    # hist[0] = ply 1: d5 = d7->d5, cap=0
    assert hist[0] == (chess.D7, chess.D5, 0), f"hist[0] at ply 2: {hist[0]}"
    # hist[1] = ply 0: e4 = e2->e4, cap=0
    assert hist[1] == (chess.E2, chess.E4, 0), f"hist[1] at ply 2: {hist[1]}"
    assert hist[2] == ABSENT, f"hist[2] at ply 2 should be absent: {hist[2]}"
    assert hist[3] == ABSENT, f"hist[3] at ply 2 should be absent: {hist[3]}"


def test_ply3_hist0_is_the_capture():
    """At ply 3 (c6 about to be played), hist[0] = exd5 (the pawn capture)."""
    by_ply = _rows_by_ply(_PGN)
    hist = by_ply[3][6]
    # hist[0] = ply 2: exd5 = e4->d5, cap=1 (PAWN)
    assert hist[0] == (chess.E4, chess.D5, chess.PAWN), (
        f"hist[0] at ply 3 should be exd5 capture, got {hist[0]}"
    )
    # hist[1] = ply 1: d5 = d7->d5, cap=0
    assert hist[1] == (chess.D7, chess.D5, 0), f"hist[1] at ply 3: {hist[1]}"
    # hist[2] = ply 0: e4 = e2->e4, cap=0
    assert hist[2] == (chess.E2, chess.E4, 0), f"hist[2] at ply 3: {hist[2]}"
    # hist[3] still absent (only 3 prior plies)
    assert hist[3] == ABSENT, f"hist[3] at ply 3 should be absent: {hist[3]}"


def test_ply4_has_all_four_slots_filled():
    """At ply 4, all 4 hist slots should be filled (plies 3,2,1,0)."""
    by_ply = _rows_by_ply(_PGN)
    hist = by_ply[4][6]
    # hist[0] = ply 3: c6 = c7->c6, cap=0
    assert hist[0] == (chess.C7, chess.C6, 0), f"hist[0] at ply 4: {hist[0]}"
    # hist[1] = ply 2: exd5 = e4->d5, cap=1
    assert hist[1] == (chess.E4, chess.D5, chess.PAWN), f"hist[1] at ply 4: {hist[1]}"
    # hist[2] = ply 1: d5 = d7->d5, cap=0
    assert hist[2] == (chess.D7, chess.D5, 0), f"hist[2] at ply 4: {hist[2]}"
    # hist[3] = ply 0: e4 = e2->e4, cap=0
    assert hist[3] == (chess.E2, chess.E4, 0), f"hist[3] at ply 4: {hist[3]}"


def test_ply5_hist_slides_window():
    """At ply 5, the deque window has moved: hist[3] = ply 1 (ply 0 dropped off)."""
    by_ply = _rows_by_ply(_PGN)
    hist = by_ply[5][6]
    # hist[0] = ply 4: Nc3 = b1->c3, cap=0
    assert hist[0] == (chess.B1, chess.C3, 0), f"hist[0] at ply 5: {hist[0]}"
    # hist[3] = ply 1: d5 = d7->d5 (ply 0 e4 has slid out of the window)
    assert hist[3] == (chess.D7, chess.D5, 0), f"hist[3] at ply 5: {hist[3]}"
    # no absent slots — all 4 filled
    for i, entry in enumerate(hist):
        assert entry != ABSENT, f"hist[{i}] at ply 5 should be present, got {entry}"


def test_skip_opening_plies_does_not_affect_hist_tracking():
    """Even when plies are skipped for sampling, hist must reflect ALL prior plies."""
    # skip the first 2 plies; first emitted ply is 2 (exd5)
    game = _parse_game(_PGN)
    _, rows = collect_candidate_positions(
        game, skip_opening_plies=2, exclude_single_legal_move=False
    )
    by_ply = {r[0]: r for r in rows}
    # ply 2 is now the first emitted row; its hist should still have e4+d5 in it
    hist = by_ply[2][6]
    assert hist[0] == (chess.D7, chess.D5, 0), f"hist[0] at ply 2 (skipped): {hist[0]}"
    assert hist[1] == (chess.E2, chess.E4, 0), f"hist[1] at ply 2 (skipped): {hist[1]}"
