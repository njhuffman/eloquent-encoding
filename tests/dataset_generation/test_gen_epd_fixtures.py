from scripts.gen_epd_fixtures import build_epd_cases

def test_cases_include_legal_and_spurious_ep():
    cases = build_epd_cases()
    by_fen = {c["fen"]: c["epd"] for c in cases}
    # legal ep: 1.e4 e6 2.e5 d5 -> exd6 legal -> epd ends with the ep target d6
    legal = "rnbqkbnr/ppp2ppp/4p3/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
    assert by_fen[legal].split()[-1] == "d6"
    # spurious ep: FEN claims e3 but no black pawn can capture -> python epd drops it to "-"
    spurious = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    assert by_fen[spurious].split()[-1] == "-"
    # every case: epd == python-chess board.epd() of the fen, and has 4 space-separated fields
    import chess
    for c in cases:
        assert c["epd"] == chess.Board(c["fen"]).epd()
        assert len(c["epd"].split()) == 4
