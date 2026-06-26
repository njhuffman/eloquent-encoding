from chess.engine import Cp, Mate
from dataset_generation.stockfish_eval import (
    CP_CLAMP, STATIC_NA, clamp_cp, parse_static_eval, score_to_cp_mate,
)

def test_clamp_cp():
    assert clamp_cp(120) == 120
    assert clamp_cp(50000) == CP_CLAMP
    assert clamp_cp(-50000) == -CP_CLAMP

def test_parse_static_eval_normal():
    assert parse_static_eval("NNUE evaluation        +0.49 (white side)") == 49
    assert parse_static_eval("NNUE evaluation        -1.23 (white side)") == -123
    assert parse_static_eval("Final evaluation: +0.00 (white side)") == 0

def test_parse_static_eval_in_check():
    assert parse_static_eval("Final evaluation: none (in check)") is None

def test_score_to_cp_mate():
    assert score_to_cp_mate(Cp(120)) == (120, 0)
    assert score_to_cp_mate(Cp(50000)) == (CP_CLAMP, 0)   # clamped
    assert score_to_cp_mate(Mate(3)) == (CP_CLAMP, 3)
    assert score_to_cp_mate(Mate(-2)) == (-CP_CLAMP, -2)
