"""Tests for PGN header prefilter and filtered game text iterator."""

from __future__ import annotations

import io

import chess

from dataset_generation.pgn_prefilter import passes_header_prefilter
from dataset_generation.recipe import Recipe, SourcePlan, StratumSpec
from dataset_generation.stream import _iter_filtered_pgn_game_texts


def _recipe_600_white_bucket() -> Recipe:
    st = StratumSpec(elo_min=1400, elo_max=1600, take_games=10, samples_per_game=1, stratum_seed=1)
    plan = SourcePlan(source="x", strata=(st,))
    return Recipe(
        name="t",
        master_seed=1,
        time_control="600+0",
        bucket_by="white",
        skip_opening_plies=0,
        exclude_single_legal_move=False,
        source_plans=(plan,),
    )


def test_prefilter_accepts_matching_headers() -> None:
    r = _recipe_600_white_bucket()
    plan = r.source_plans[0]
    acc = [0]
    assert passes_header_prefilter(
        parsed_tc="600+0",
        white_elo=1500,
        black_elo=1200,
        recipe=r,
        plan=plan,
        accepted=acc,
    )


def test_prefilter_rejects_wrong_time_control() -> None:
    r = _recipe_600_white_bucket()
    plan = r.source_plans[0]
    acc = [0]
    assert not passes_header_prefilter(
        parsed_tc="60+0",
        white_elo=1500,
        black_elo=1200,
        recipe=r,
        plan=plan,
        accepted=acc,
    )


def test_prefilter_rejects_out_of_range_elo() -> None:
    r = _recipe_600_white_bucket()
    plan = r.source_plans[0]
    acc = [0]
    assert not passes_header_prefilter(
        parsed_tc="600+0",
        white_elo=800,
        black_elo=1200,
        recipe=r,
        plan=plan,
        accepted=acc,
    )


def test_stream_yields_only_prefiltered_games() -> None:
    r = _recipe_600_white_bucket()
    plan = r.source_plans[0]
    acc = [0]
    pgn = """[Event "g600"]
[TimeControl "600+0"]
[WhiteElo "1500"]
[BlackElo "1200"]

1. e4 e5
"""
    pgn += """[Event "g60"]
[TimeControl "60+0"]
[WhiteElo "1500"]
[BlackElo "1200"]

1. e4 e5
"""
    games = list(_iter_filtered_pgn_game_texts(io.StringIO(pgn), recipe=r, plan=plan, accepted=acc))
    assert len(games) == 1
    g = chess.pgn.read_game(io.StringIO(games[0]))
    assert g is not None
    assert g.headers.get("TimeControl") == "600+0"


def test_stream_stops_immediately_if_quotas_already_satisfied() -> None:
    """First line read sees full quotas → StopIteration before buffering any game."""
    r = _recipe_600_white_bucket()
    plan = r.source_plans[0]
    acc = [10]  # take_games is 10 → nothing left to fill
    pgn = """[Event "tail_only"]
[TimeControl "600+0"]
[WhiteElo "1500"]
[BlackElo "1200"]

1. e4 e5
"""
    games = list(_iter_filtered_pgn_game_texts(io.StringIO(pgn), recipe=r, plan=plan, accepted=acc))
    assert games == []


def test_iterator_stops_after_quotas_met_between_yields() -> None:
    """After the parent fills quotas, the next pull must not scan the rest of the file."""
    r = _recipe_600_white_bucket()
    plan = r.source_plans[0]
    acc = [0]
    long_tail = "\n" + "\n".join(f"{i}. e4 e5 2. Nf3 Nc6" for i in range(3, 500))
    pgn = """[Event "one"]
[TimeControl "600+0"]
[WhiteElo "1500"]
[BlackElo "1200"]

1. e4 e5
""" + """[Event "two"]
[TimeControl "600+0"]
[WhiteElo "1500"]
[BlackElo "1200"]

1. e4 e5
""" + long_tail

    it = _iter_filtered_pgn_game_texts(io.StringIO(pgn), recipe=r, plan=plan, accepted=acc)
    first = next(it)
    assert "600+0" in first and '[Event "one"]' in first
    acc[0] = 10
    try:
        second = next(it)
    except StopIteration:
        pass
    else:
        raise AssertionError(f"expected StopIteration, got second chunk len={len(second)}")


def test_board_at_ply_matches_fen_path() -> None:
    from dataset_generation.candidate_collect import board_at_ply

    pgn = """[Event "x"]
[TimeControl "600+0"]
[WhiteElo "1500"]
[BlackElo "1500"]

1. e4 e5 2. Nf3 Nc6
"""
    g = chess.pgn.read_game(io.StringIO(pgn))
    assert g is not None
    mainline = list(g.mainline_moves())
    # ply 2 = before third half-move (after 1.e4 e5)
    b = board_at_ply(mainline, 2)
    expected = chess.Board(
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    )
    assert b.fen() == expected.fen()
