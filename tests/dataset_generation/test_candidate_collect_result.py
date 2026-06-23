import io
import chess.pgn
from dataset_generation.candidate_collect import collect_candidate_positions

_PGN = """[White "a"]
[Black "b"]
[WhiteElo "1500"]
[BlackElo "1600"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0
"""

def _game():
    return chess.pgn.read_game(io.StringIO(_PGN))

def test_rows_carry_opp_elo_and_result_from_stm_perspective():
    _, rows = collect_candidate_positions(_game(), skip_opening_plies=0, exclude_single_legal_move=False)
    assert rows, "expected candidate rows"
    # White (stm=0) won -> result 2, opp_elo=1600; Black (stm=1) lost -> result 0, opp_elo=1500
    for ply, stm, elo, opp, result, move in rows:
        if stm == 0:
            assert elo == 1500 and opp == 1600 and result == 2
        else:
            assert elo == 1600 and opp == 1500 and result == 0

def test_unterminated_game_dropped():
    pgn = _PGN.replace('[Result "1-0"]', '[Result "*"]').replace(" 1-0", " *")
    _, rows = collect_candidate_positions(chess.pgn.read_game(io.StringIO(pgn)),
                                          skip_opening_plies=0, exclude_single_legal_move=False)
    assert rows == []
