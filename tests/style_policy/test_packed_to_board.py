import chess
from style_policy.board_encode import board_to_packed, packed_to_board

def test_packed_to_board_roundtrip():
    fens = [
        chess.STARTING_FEN,
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "rnbqkbnr/pp1ppppp/8/2pP4/8/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 3",  # ep
        "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b Kq - 0 1",                          # partial castling
    ]
    for fen in fens:
        b = chess.Board(fen)
        rt = packed_to_board(board_to_packed(b))
        assert rt.board_fen() == b.board_fen()
        assert rt.turn == b.turn
        assert rt.castling_rights == b.castling_rights
        assert rt.ep_square == b.ep_square
