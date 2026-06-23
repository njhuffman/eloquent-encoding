import chess
from style_policy.opening_book import BookBuilder

def _ucis(*sans):
    b = chess.Board(); out = []
    for s in sans:
        m = b.parse_san(s); out.append(m); b.push(m)
    return out

def test_counts_and_transposition_merge():
    bld = BookBuilder(n_plies=6)
    bld.add_game(1800, _ucis("e4", "e5", "Nf3", "Nc6", "Bb5"))
    bld.add_game(1850, _ucis("Nf3", "Nc6", "e4", "e5", "Bb5"))
    books = bld.finalize(min_support=0.0)
    bk = books[1800]
    assert bk.total_games == 2
    start = chess.Board().epd()
    assert bk.positions[start]["n"] == 2
    assert bk.positions[start]["moves"] == {"e2e4": 1, "g1f3": 1}
    # the two move orders transpose after 4 plies; the 5th move (Bb5) is recorded
    # from that shared position under one EPD key -> pooled
    merge = chess.Board()
    for m in _ucis("e4", "e5", "Nf3", "Nc6"):
        merge.push(m)
    assert bk.positions[merge.epd()]["n"] == 2
    assert bk.positions[merge.epd()]["moves"] == {"f1b5": 2}
    # invariant: every recorded position has sum(move counts) == n (no move-less entries)
    for e in bk.positions.values():
        assert sum(e["moves"].values()) == e["n"]

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
