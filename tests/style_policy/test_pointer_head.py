import numpy as np, chess
from style_policy.board_encode import legal_move_matrix, legal_from_u64

def test_legal_move_matrix_matches_board():
    board = chess.Board()
    m = legal_move_matrix(board)
    pairs = {(mv.from_square, mv.to_square) for mv in board.legal_moves}
    assert m.sum() == len(pairs)
    for (fr, to) in pairs:
        assert m[fr, to]
    froms = {i for i in range(64) if m[i].any()}
    lf = legal_from_u64(board)
    assert froms == {i for i in range(64) if (lf >> i) & 1}
