import chess
from style_policy.opening_book import BookBuilder

def _ucis(*sans):
    b = chess.Board(); out = []
    for s in sans:
        m = b.parse_san(s); out.append(m); b.push(m)
    return out

def test_counts_and_transposition_merge():
    bld = BookBuilder(n_plies=6)
    # Two move orders reaching the same position after 1.e4 e5 2.Nf3 / 1.Nf3 e5 2.e4
    bld.add_game(1800, _ucis("e4", "e5", "Nf3"))
    bld.add_game(1850, _ucis("Nf3", "e5", "e4"))
    books = bld.finalize(min_support=0.0)
    bk = books[1800]
    assert bk.total_games == 2
    start = chess.Board().epd()
    # start position seen by both games; its move counts split e4 vs Nf3
    assert bk.positions[start]["n"] == 2
    assert bk.positions[start]["moves"] == {"e2e4": 1, "g1f3": 1}
    # after 1.e4 e5 2.Nf3 and 1.Nf3 e5 2.e4 the positions transpose -> pooled under one EPD
    b1 = chess.Board(); [b1.push(m) for m in _ucis("e4", "e5", "Nf3")]
    assert b1.epd() in bk.positions  # reached by game 1's 3rd ply and game 2's final position

def test_out_of_range_white_elo_ignored():
    bld = BookBuilder(n_plies=4)
    bld.add_game(2500, _ucis("e4", "e5"))   # White elo out of [1000,1999]
    assert bld.finalize(0.0) == {}

def test_prune_drops_low_support():
    bld = BookBuilder(n_plies=2)
    for _ in range(100):
        bld.add_game(1500, _ucis("e4", "e5"))
    for _ in range(1):
        bld.add_game(1500, _ucis("d4", "d5"))   # 1/101 ~ 0.0099 support at start? no—start seen by all 101
    books = bld.finalize(min_support=0.02)
    bk = books[1500]
    start = chess.Board().epd()
    # start position support = 101/101 = 1.0 kept; its rare move d2d4 stays in the move dict
    assert start in bk.positions
    # the position after 1.d4 (seen once, support 1/101 ~0.0099 < 0.02) is pruned
    b = chess.Board(); b.push(chess.Move.from_uci("d2d4"))
    assert b.epd() not in bk.positions
