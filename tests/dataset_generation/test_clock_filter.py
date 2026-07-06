import io

import chess.pgn

from dataset_generation.candidate_collect import collect_candidate_positions

# 8-ply game with explicit [%clk h:mm:ss] comments. clocks[i] = remaining clock
# (seconds) after ply i is played:
#   ply0 white e4  -> 0:02:30 = 150
#   ply1 black e5  -> 0:02:40 = 160
#   ply2 white Nf3 -> 0:00:20 =  20
#   ply3 black Nc6 -> 0:02:30 = 150
#   ply4 white Bb5 -> 0:00:10 =  10
#   ply5 black a6  -> 0:02:00 = 120
#   ply6 white Ba4 -> 0:01:00 =  60
#   ply7 black Nf6 -> 0:00:50 =  50
_PGN = """[White "a"]
[Black "b"]
[WhiteElo "1500"]
[BlackElo "1600"]
[Result "1-0"]
[TimeControl "180+0"]

1. e4 {[%clk 0:02:30]} e5 {[%clk 0:02:40]} 2. Nf3 {[%clk 0:00:20]} Nc6 {[%clk 0:02:30]} 3. Bb5 {[%clk 0:00:10]} a6 {[%clk 0:02:00]} 4. Ba4 {[%clk 0:01:00]} Nf6 {[%clk 0:00:50]} 1-0
"""


def _game():
    return chess.pgn.read_game(io.StringIO(_PGN))


def _plies(rows):
    return {r[0] for r in rows}


def test_disabled_filter_keeps_all_plies():
    _, rows = collect_candidate_positions(
        _game(),
        skip_opening_plies=0,
        exclude_single_legal_move=False,
        min_clock_seconds=0,
        base_seconds=180,
    )
    assert _plies(rows) == set(range(8))


def test_min_clock_drops_time_pressure_decisions():
    # Decision clock for ply p = clocks[p-2] (mover's own previous move), or base (180)
    # for their first move (p < 2). With min_clock_seconds=30:
    #   p=0 base 180  keep      p=1 base 180  keep
    #   p=2 clocks[0]=150 keep  p=3 clocks[1]=160 keep
    #   p=4 clocks[2]= 20 DROP  p=5 clocks[3]=150 keep
    #   p=6 clocks[4]= 10 DROP  p=7 clocks[5]=120 keep
    _, rows = collect_candidate_positions(
        _game(),
        skip_opening_plies=0,
        exclude_single_legal_move=False,
        min_clock_seconds=30,
        base_seconds=180,
    )
    assert _plies(rows) == {0, 1, 2, 3, 5, 7}


def test_missing_base_keeps_first_moves():
    # base_seconds=None -> p<2 decision clock unknown -> kept (conservative).
    _, rows = collect_candidate_positions(
        _game(),
        skip_opening_plies=0,
        exclude_single_legal_move=False,
        min_clock_seconds=30,
        base_seconds=None,
    )
    # Same drops as above (those depend on clocks, not base); first moves still kept.
    assert {0, 1}.issubset(_plies(rows))
    assert _plies(rows) == {0, 1, 2, 3, 5, 7}
