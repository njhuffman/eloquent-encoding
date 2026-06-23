import io, zstandard, chess
from pathlib import Path
from style_policy.opening_book import OpeningBook
from scripts.build_opening_book import build

_GAME = """[Event "x"]
[White "a"]
[Black "b"]
[WhiteElo "1850"]
[BlackElo "1840"]
[TimeControl "600+0"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0

"""

def test_build_produces_band_file(tmp_path):
    raw = (_GAME * 5).encode()
    zpath = tmp_path / "src.pgn.zst"
    zpath.write_bytes(zstandard.ZstdCompressor().compress(raw))
    out = tmp_path / "book"
    bands = build(zpath, out, n_plies=6, per_band_target=5, min_support=0.0)
    assert bands == [1800]
    bk = OpeningBook.load(out / "band_1800.json")
    assert bk.total_games == 5
    assert bk.positions[chess.Board().epd()]["moves"] == {"e2e4": 5}
