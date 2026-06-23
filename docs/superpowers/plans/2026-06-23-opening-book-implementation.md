# Opening Book Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A per-elo-band, PGN-derived statistical opening book that plays human-frequency opening moves until a position's representation drops below a threshold, then hands off to the model policy.

**Architecture:** An offline builder tallies the first N plies of band-bucketed games into position-keyed (EPD) frequency tables; at play time `PolicyBot` looks up the current position in its band's book and samples a move ∝ human frequency while support ≥ threshold, otherwise falls through to the existing model path.

**Tech Stack:** Python, python-chess, zstandard; existing `dataset_generation.stream` + `style_policy.play`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-06-23-opening-book-design.md`.
- **Position key = `board.epd()`** (piece placement + side-to-move + castling + ep) — merges transpositions.
- **Bucket games by White elo** (matches the dataset's `bucket_by: white`); a game is used only if White elo ∈ [1000, 1999]. Both sides' first-N-ply moves are tallied into that band (EPD's side-to-move separates the colors).
- **N = 24 plies** tallied per game.
- **support% = position.n / band.total_games** (the exit knob). **Move sampling = conditional** `move_count / position.n`, ∝ counts, **ignoring play-temperature**.
- **Build-prune:** keep positions with `support% ≥ 0.001` (0.1%). **Runtime threshold** default `0.01` (1%), must be ≥ 0.001.
- **elo → band** (play): `clamp((elo // 100) * 100, 1000, 1900)`.
- Book files: per band `band_<lower>.json` under `/mnt/eloquence_bulk/databases/opening_book/` (data dir, not the repo). Format: `{"band": <int>, "total_games": <int>, "positions": {epd: {"n": int, "moves": {uci: int}}}}`.
- Tests/python run in the container: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 pytest <path> -q'` (pytest, NOT `python -m pytest`).
- Reuse `dataset_generation.stream.iter_pgn_games_from_zstd_binary`; do not add the recipe machinery.

---

## File Structure

- `style_policy/opening_book.py` (create) — `elo_to_band`, `OpeningBook` (play-side: load/save/for_elo/lookup), `BookBuilder` (build-side: add_game/finalize/save_all).
- `scripts/build_opening_book.py` (create) — stream a `.pgn.zst`, filter by TC + White-elo band, feed `BookBuilder`, save per-band files.
- `style_policy/play.py` (modify) — `PolicyBot` accepts an optional `opening_book` + `book_threshold`; `choose_move` consults the book first.
- Tests under `tests/style_policy/`.

---

### Task 1: `elo_to_band` + `OpeningBook` (play-side)

**Files:**
- Create: `style_policy/opening_book.py`
- Test: `tests/style_policy/test_opening_book.py`

**Interfaces:**
- Produces:
  - `elo_to_band(elo: int) -> int` = `clamp((elo // 100) * 100, 1000, 1900)`.
  - `OpeningBook(total_games: int, positions: dict[str, dict])` with attrs `total_games`, `positions` (`{epd: {"n": int, "moves": {uci: int}}}`).
  - `OpeningBook.lookup(board: chess.Board, threshold: float, rand: random.Random) -> chess.Move | None`: if `board.epd()` is in `positions` and `n/total_games ≥ threshold`, sample a **legal** move ∝ conditional `moves` counts using `rand`; else `None`.
  - `OpeningBook.save(path)` / `OpeningBook.load(path) -> OpeningBook` (JSON).
  - `OpeningBook.for_elo(book_dir, elo) -> OpeningBook | None` (maps elo→band, loads `band_<band>.json`, returns `None` if the file is absent).

- [ ] **Step 1: Write the failing test**

`tests/style_policy/test_opening_book.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 pytest tests/style_policy/test_opening_book.py -q'`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

`style_policy/opening_book.py`:

```python
"""Per-elo-band statistical opening book (PGN-derived). Play-side OpeningBook +
build-side BookBuilder. Positions keyed by board.epd() (transposition-merged)."""
from __future__ import annotations
import json
import random
from pathlib import Path
import chess


def elo_to_band(elo: int) -> int:
    """Lower bound of the 100-wide band, clamped to the built range [1000, 1900]."""
    return max(1000, min(1900, (int(elo) // 100) * 100))


class OpeningBook:
    def __init__(self, total_games: int, positions: dict[str, dict]):
        self.total_games = int(total_games)
        self.positions = positions  # {epd: {"n": int, "moves": {uci: int}}}

    def lookup(self, board: chess.Board, threshold: float, rand: random.Random) -> chess.Move | None:
        entry = self.positions.get(board.epd())
        if entry is None or self.total_games <= 0:
            return None
        if entry["n"] / self.total_games < threshold:
            return None
        legal = {m.uci(): m for m in board.legal_moves}
        items = [(legal[u], c) for u, c in entry["moves"].items() if u in legal]
        if not items:
            return None
        total = sum(c for _, c in items)
        r = rand.random() * total
        acc = 0.0
        for mv, c in items:
            acc += c
            if r <= acc:
                return mv
        return items[-1][0]

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"total_games": self.total_games, "positions": self.positions}))

    @classmethod
    def load(cls, path) -> "OpeningBook":
        d = json.loads(Path(path).read_text())
        return cls(d["total_games"], d["positions"])

    @classmethod
    def for_elo(cls, book_dir, elo: int) -> "OpeningBook | None":
        p = Path(book_dir) / f"band_{elo_to_band(elo)}.json"
        return cls.load(p) if p.exists() else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 pytest tests/style_policy/test_opening_book.py -q'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add style_policy/opening_book.py tests/style_policy/test_opening_book.py
git commit -m "feat: OpeningBook (elo_to_band + position-keyed lookup/load/save)"
```

---

### Task 2: `BookBuilder` (build-side)

**Files:**
- Modify: `style_policy/opening_book.py`
- Test: `tests/style_policy/test_book_builder.py`

**Interfaces:**
- Consumes: `elo_to_band`, `OpeningBook` (Task 1).
- Produces:
  - `BookBuilder(n_plies: int = 24)` accumulating per-band counts.
  - `add_game(white_elo: int, moves: list[chess.Move])`: if `1000 ≤ white_elo ≤ 1999`, replay the first `n_plies` of `moves`; for each position (before its move) increment `band.positions[epd]["n"]` and `["moves"][uci]`; increment `band.total_games`. Games with White elo out of range are ignored.
  - `finalize(min_support: float = 0.001) -> dict[int, OpeningBook]`: per band, drop positions with `n/total_games < min_support`; return `{band: OpeningBook}`.
  - `save_all(out_dir, min_support: float = 0.001) -> list[int]`: finalize and write `band_<band>.json` per non-empty band; return the bands written.

- [ ] **Step 1: Write the failing test**

`tests/style_policy/test_book_builder.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 pytest tests/style_policy/test_book_builder.py -q'`
Expected: FAIL — `cannot import name 'BookBuilder'`.

- [ ] **Step 3: Implement**

Append to `style_policy/opening_book.py`:

```python
class BookBuilder:
    def __init__(self, n_plies: int = 24):
        self.n_plies = int(n_plies)
        self._bands: dict[int, dict[str, dict]] = {}
        self._totals: dict[int, int] = {}

    def add_game(self, white_elo: int, moves: list[chess.Move]) -> None:
        if not (1000 <= int(white_elo) <= 1999):
            return
        band = (int(white_elo) // 100) * 100
        positions = self._bands.setdefault(band, {})
        self._totals[band] = self._totals.get(band, 0) + 1
        board = chess.Board()
        for i, mv in enumerate(moves):
            if i >= self.n_plies:
                break
            entry = positions.setdefault(board.epd(), {"n": 0, "moves": {}})
            entry["n"] += 1
            u = mv.uci()
            entry["moves"][u] = entry["moves"].get(u, 0) + 1
            board.push(mv)

    def finalize(self, min_support: float = 0.001) -> dict[int, OpeningBook]:
        out: dict[int, OpeningBook] = {}
        for band, positions in self._bands.items():
            total = self._totals[band]
            kept = {epd: e for epd, e in positions.items() if e["n"] / total >= min_support}
            out[band] = OpeningBook(total, kept)
        return out

    def save_all(self, out_dir, min_support: float = 0.001) -> list[int]:
        from pathlib import Path
        out_dir = Path(out_dir)
        written = []
        for band, book in self.finalize(min_support).items():
            if not book.positions:
                continue
            data = {"band": band, "total_games": book.total_games, "positions": book.positions}
            (out_dir).mkdir(parents=True, exist_ok=True)
            (out_dir / f"band_{band}.json").write_text(__import__("json").dumps(data))
            written.append(band)
        return sorted(written)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 pytest tests/style_policy/test_book_builder.py -q'`
Expected: PASS. (If the transposition assertion is sensitive to EPD ep-square nuances, confirm both move orders produce identical `board.epd()` at the merge point; they should, since neither has an ep-capturable pawn.)

- [ ] **Step 5: Commit**

```bash
git add style_policy/opening_book.py tests/style_policy/test_book_builder.py
git commit -m "feat: BookBuilder (per-band tally, prune, save)"
```

---

### Task 3: Build script

**Files:**
- Create: `scripts/build_opening_book.py`
- Test: `tests/style_policy/test_build_opening_book.py`

**Interfaces:**
- Consumes: `BookBuilder` (Task 2), `dataset_generation.stream.iter_pgn_games_from_zstd_binary`.
- Produces: `build(pgn_zst_path, out_dir, *, n_plies=24, per_band_target=100000, time_control="600+0", min_support=0.001) -> list[int]` — streams games, keeps `TimeControl==time_control` games with a valid White elo, feeds `BookBuilder` until every covered band reaches `per_band_target` (or the stream ends), then `save_all`. Returns bands written.

- [ ] **Step 1: Write the failing test (tiny e2e on a synthetic zst)**

`tests/style_policy/test_build_opening_book.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 pytest tests/style_policy/test_build_opening_book.py -q'`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

`scripts/build_opening_book.py`:

```python
#!/usr/bin/env python3
"""Build per-elo-band opening books from a Lichess .pgn.zst (bucket by White elo)."""
from __future__ import annotations
import argparse, io
from pathlib import Path
import chess.pgn
from tqdm import tqdm
from dataset_generation.stream import iter_pgn_games_from_zstd_binary
from style_policy.opening_book import BookBuilder

_BANDS = [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]


def _white_elo(headers) -> int | None:
    raw = headers.get("WhiteElo")
    if raw is None or raw == "?":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def build(pgn_zst_path, out_dir, *, n_plies: int = 24, per_band_target: int = 100000,
          time_control: str = "600+0", min_support: float = 0.001) -> list[int]:
    bld = BookBuilder(n_plies=n_plies)
    counts = {b: 0 for b in _BANDS}
    with open(pgn_zst_path, "rb") as raw:
        for text in tqdm(iter_pgn_games_from_zstd_binary(raw), unit=" games"):
            if all(counts[b] >= per_band_target for b in _BANDS):
                break
            game = chess.pgn.read_game(io.StringIO(text))
            if game is None or game.headers.get("TimeControl") != time_control:
                continue
            we = _white_elo(game.headers)
            if we is None or not (1000 <= we <= 1999):
                continue
            band = (we // 100) * 100
            if counts[band] >= per_band_target:
                continue
            moves = list(game.mainline_moves())
            bld.add_game(we, moves)
            counts[band] += 1
    return bld.save_all(out_dir, min_support=min_support)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-plies", type=int, default=24)
    ap.add_argument("--per-band-target", type=int, default=100000)
    ap.add_argument("--min-support", type=float, default=0.001)
    args = ap.parse_args()
    bands = build(args.pgn, args.out, n_plies=args.n_plies,
                  per_band_target=args.per_band_target, min_support=args.min_support)
    print("wrote bands:", bands)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 pytest tests/style_policy/test_build_opening_book.py -q'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_opening_book.py tests/style_policy/test_build_opening_book.py
git commit -m "feat: opening-book build script (stream + band tally)"
```

---

### Task 4: `PolicyBot` opening-book integration

**Files:**
- Modify: `style_policy/play.py`
- Test: `tests/style_policy/test_policybot_book.py`

**Interfaces:**
- Consumes: `OpeningBook` (Task 1).
- Produces: `PolicyBot.__init__` accepts optional `opening_book: OpeningBook | None = None` and `book_threshold: float = 0.01`, and creates `self._book_rng = random.Random(seed if seed is not None else 0)`. `choose_move` first tries `self.opening_book.lookup(board, self.book_threshold, self._book_rng)`; if it returns a move, return it; else the existing model path runs unchanged.

- [ ] **Step 1: Write the failing test**

`tests/style_policy/test_policybot_book.py`:

```python
import random, chess
from style_policy.opening_book import OpeningBook

class _StubModel:
    def __getattr__(self, _):  # any model attr access -> should NOT happen when book hits
        raise AssertionError("model path used while opening book should have answered")

def test_book_move_short_circuits_model(monkeypatch):
    from style_policy import play
    # Build a bot without loading a real checkpoint: bypass __init__ via __new__
    bot = play.PolicyBot.__new__(play.PolicyBot)
    bot.opening_book = OpeningBook(total_games=100, positions={
        chess.Board().epd(): {"n": 100, "moves": {"e2e4": 100}}})
    bot.book_threshold = 0.01
    bot._book_rng = random.Random(0)
    bot.model = _StubModel()  # explodes if touched
    mv = play.PolicyBot.choose_move(bot, chess.Board())
    assert mv == chess.Move.from_uci("e2e4")

def test_no_book_falls_through(monkeypatch):
    from style_policy import play
    bot = play.PolicyBot.__new__(play.PolicyBot)
    bot.opening_book = None
    bot.book_threshold = 0.01
    bot._book_rng = random.Random(0)
    called = {"hit": False}
    # stub the model path: replace choose_move's model use by monkeypatching the method's tail.
    # Simplest: give a book that returns None and assert lookup path doesn't crash; then
    # verify the bot attempts the model path by raising a sentinel from a stubbed encode.
    class _M:
        def encode(self, *a, **k):
            called["hit"] = True
            raise RuntimeError("model-path-reached")
    bot.model = _M()
    try:
        play.PolicyBot.choose_move(bot, chess.Board())
    except RuntimeError as e:
        assert "model-path-reached" in str(e)
    assert called["hit"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 pytest tests/style_policy/test_policybot_book.py -q'`
Expected: FAIL — `choose_move` ignores the book / `opening_book` attr unused.

- [ ] **Step 3: Implement**

In `style_policy/play.py`:

(a) add `import random` at the top (with the other imports).

(b) extend `PolicyBot.__init__` signature and body — add params and the book RNG (place alongside the existing `self.gen` setup):

```python
    def __init__(self, checkpoint_path, elo, *, device="cpu", temperature=1.0, seed=None,
                 opening_book=None, book_threshold: float = 0.01):
        ...  # existing body unchanged ...
        self.opening_book = opening_book
        self.book_threshold = float(book_threshold)
        self._book_rng = random.Random(seed if seed is not None else 0)
```

(c) at the very top of `choose_move`, before the model work:

```python
    @torch.no_grad()
    def choose_move(self, board: chess.Board) -> chess.Move:
        if self.opening_book is not None:
            mv = self.opening_book.lookup(board, self.book_threshold, self._book_rng)
            if mv is not None:
                return mv
        # ---- existing model path unchanged below ----
        pk = torch.from_numpy(board_to_packed(board)[None]).to(self.device)
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 pytest tests/style_policy/test_policybot_book.py -q'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add style_policy/play.py tests/style_policy/test_policybot_book.py
git commit -m "feat: PolicyBot consults the opening book before the model"
```

---

### Task 5: Build the real per-band books + sanity (controller-run; long CPU job)

**Files:** none (produces `/mnt/eloquence_bulk/databases/opening_book/band_*.json`).

Not a TDD task — runs the Task-3 script on real data. Controller launches/monitors.

- [ ] **Step 1: Build**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 python scripts/build_opening_book.py --pgn /mnt/eloquence_bulk/databases/lichess_db_standard_rated_2025-01_tc_600_0.pgn.zst --out /mnt/eloquence_bulk/databases/opening_book --per-band-target 100000 >/tmp/bookbuild.log 2>&1; echo EXIT=$?'`
Expected: `EXIT=0`, `wrote bands: [1000, 1100, ..., 1900]`.

- [ ] **Step 2: Sanity-check**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && python -c "
import chess, random
from style_policy.opening_book import OpeningBook
bk = OpeningBook.for_elo(\"/mnt/eloquence_bulk/databases/opening_book\", 1800)
start = chess.Board().epd()
mv = bk.positions[start][\"moves\"]
top = sorted(mv.items(), key=lambda kv: -kv[1])[:5]
print(\"total_games\", bk.total_games, \"start top moves\", top)
print(\"n_positions\", len(bk.positions))
"'`
Expected: start-position top moves are sane (e2e4 / d2d4 / g1f3 dominate); `total_games` ≈ 100000; a reasonable `n_positions`.

- [ ] **Step 3: Record + commit a DEVLOG note**

Append the build command, bands written, per-band `total_games` / `n_positions`, and the 1800 start-position top moves to `docs/DEVLOG.md`; commit.

---

## Self-Review

**Spec coverage:** PGN-derived (Tasks 2,3,5); per-band bucket-by-White (Task 2/3); N=24 (constraint + Tasks 2,3); EPD key + transposition merge (Task 2, tested); support% exit + conditional ∝ sampling ignoring temperature (Task 1 `lookup`, tested); build-prune 0.1% (Task 2 `finalize`); runtime threshold 1% default (Task 4); elo→band clamp (Task 1, tested); files under the data dir (Task 3/5); PolicyBot integration with model fall-through (Task 4, tested). All covered.

**Placeholder scan:** every code step has complete code; commands have expected output. Task 5 is an explicit long controller-run job with exact commands.

**Type consistency:** `OpeningBook(total_games, positions)` and the `{epd: {"n", "moves": {uci:count}}}` shape are identical across Tasks 1, 2, 3. `elo_to_band` (Task 1) used by `for_elo` (1) and bucket logic (2,3 use the inline `(elo//100)*100` after the in-range check, consistent with the clamp for in-range values). `lookup(board, threshold, rand)` signature matches Task 4's call. `BookBuilder(n_plies)` / `add_game(white_elo, moves)` / `save_all(out_dir, min_support)` consistent between Tasks 2 and 3.
