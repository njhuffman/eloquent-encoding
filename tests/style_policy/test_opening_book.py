import random
import chess
from style_policy.opening_book import elo_to_band, OpeningBook

def test_elo_to_band_clamps():
    assert elo_to_band(1850) == 1800
    assert elo_to_band(1000) == 1000
    assert elo_to_band(600) == 1000   # clamp low
    assert elo_to_band(2400) == 1900  # clamp high

def _book():
    start = chess.Board().epd()
    return OpeningBook(total_games=1000, positions={
        start: {"n": 900, "moves": {"e2e4": 600, "d2d4": 300}},          # support 0.9
        "rare": {"n": 5, "moves": {"e2e4": 5}},                          # support 0.005
    })

def test_lookup_returns_book_move_above_threshold():
    b = _book(); board = chess.Board()
    mv = b.lookup(board, threshold=0.01, rand=random.Random(0))
    assert mv in (chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4"))
    assert mv in board.legal_moves

def test_lookup_none_below_threshold_or_unknown():
    b = _book(); board = chess.Board()
    # an EPD not in the book
    empty = OpeningBook(total_games=1000, positions={})
    assert empty.lookup(board, 0.01, random.Random(0)) is None
    # known but below threshold: a book whose only entry is the start at support 0.005
    low = OpeningBook(total_games=1000, positions={board.epd(): {"n": 5, "moves": {"e2e4": 5}}})
    assert low.lookup(board, 0.01, random.Random(0)) is None

def test_lookup_seeded_is_deterministic_and_distribution_holds():
    b = _book(); board = chess.Board()
    a = b.lookup(board, 0.01, random.Random(42))
    c = b.lookup(board, 0.01, random.Random(42))
    assert a == c
    # 600:300 -> e2e4 should dominate over many draws
    n_e4 = sum(b.lookup(chess.Board(), 0.01, random.Random(s)) == chess.Move.from_uci("e2e4")
               for s in range(200))
    assert n_e4 > 120  # ~2/3 expected

def test_save_load_roundtrip(tmp_path):
    b = _book(); p = tmp_path / "band_1800.json"; b.save(p)
    r = OpeningBook.load(p)
    assert r.total_games == 1000 and r.positions[chess.Board().epd()]["moves"]["e2e4"] == 600

def test_for_elo_loads_band_file(tmp_path):
    _book().save(tmp_path / "band_1800.json")
    assert OpeningBook.for_elo(tmp_path, 1850).total_games == 1000
    assert OpeningBook.for_elo(tmp_path, 1250) is None  # no band_1200.json
