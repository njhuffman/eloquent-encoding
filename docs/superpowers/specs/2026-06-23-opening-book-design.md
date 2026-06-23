# Opening book (per-elo-band, PGN-derived) — design

**Status:** approved design; precedes the implementation plan.

## Goal

Give the bot human-realistic openings (and put the resulting middlegame in the model's
training distribution) via a per-elo-band statistical opening book derived from the project's
own PGNs. While the current position is well-represented in the band's games, the bot samples
its move from the human frequency distribution there; once representation drops below a
threshold it hands off to the existing model policy. This removes the out-of-distribution
move-1 behavior (e.g., the early king-walk at 1800) and makes downstream bot-strength/blunder
evals meaningful (no move-1 OOD noise).

## Why a frequency book (not learning openings into the model)

Openings are a frequency/memorization task with huge branching and low per-position counts —
a count table is the right tool. The model is weak at move 1 precisely because it tries to
learn (from sparse, skipped data) what a lookup does perfectly. The book owns the opening; the
model owns the middlegame it was trained on.

## Decisions

1. **Source:** ~100k games/band from `lichess_db_standard_rated_2025-01_tc_600_0.pgn.zst`
   (same distribution as the 16M training set). 10 bands: 1000–1099 … 1900–1999.
2. **Depth:** tally the first **N = 24 plies** (12 full moves) of each game — well past the
   model's `skip_opening_plies` (4).
3. **Position key:** `board.epd()` (piece placement + side-to-move + castling + ep) — **merges
   transpositions** (same position via different move orders pools counts).
4. **Two normalizations:**
   - `support% = position.games_through / band.total_games` — the *exit knob* (how common the
     position is among the band's games).
   - move choice = **conditional** `move_count / position.games_through`; in-book moves are
     sampled ∝ these conditional counts (true human opening variety, **ignoring the bot's
     play-temperature**).
5. **Build-time prune:** keep only positions with `support% ≥ 0.1%` (the lowest runtime
   threshold we'd use) — discards the deep unique tail, bounds file size.
6. **Runtime threshold:** a `PolicyBot` config knob, default **1%**; must be ≥ the 0.1% build
   floor.
7. **elo → band:** `floor(elo/100)·100`, clamped to `[1000 … 1900]` for out-of-range elos
   (the bot's elo slider spans 600–2400; bands only cover 1000–1999).

## Components

### 1. Build script — `scripts/build_opening_book.py`
- Stream the band's games (reuse `dataset_generation/stream.py` filtering: time control `600+0`,
  the band's elo range via the existing header-elo logic), up to ~100k games/band.
- For each game, replay the first N=24 plies; at each position before a move, tally
  `book[band][epd].games_through += 1` and `book[band][epd].moves[uci] += 1`. (Count each game
  once per distinct position it passes through.)
- Track `band.total_games`.
- Prune positions with `support% < 0.1%`.
- Write one JSON file per band (or a single JSON keyed by band) under
  `/mnt/eloquence_bulk/databases/opening_book/` (data dir, not the repo): per band
  `{total_games, positions: {epd: {n, moves: {uci: count}}}}`.

### 2. `OpeningBook` — `style_policy/opening_book.py`
- `OpeningBook.load(path, band) -> OpeningBook` (loads one band's table).
- `lookup(board, threshold, rng) -> chess.Move | None`: compute `board.epd()`; if present and
  `n / total_games ≥ threshold`, sample a move ∝ conditional `moves` counts (restricted to
  legal moves as a safety check) using `rng`; else return `None` (caller falls through to the
  model).
- Pure, testable, no model dependency.

### 3. `PolicyBot` integration — `style_policy/play.py`
- New optional config: `opening_book` (an `OpeningBook` or its path + band) and
  `book_threshold` (default 0.01). `band` derived from the bot's elo via the mapping above.
- In `choose_move`: if a book is configured, try `book.lookup(board, threshold, rng)`; if it
  returns a move, play it (it's a real human move from this position, so legal); otherwise fall
  through to the existing model path unchanged.
- Reuses the bot's existing `gen` RNG for reproducibility.

## Play-time flow

```
choose_move(board):
  if book and (mv := book.lookup(board, threshold, rng)) is not None:
      return mv                 # in-book: human-frequency move
  return <existing model path>  # out of book: model owns it
```

The handoff at `support% < threshold` lands where the book's data thins — which is also where
the model's training data thins — so the middlegame starts in-distribution. Raising the
threshold hands off sooner (more model); lowering goes deeper into human openings.

## Testing

- **Build (unit):** build a tiny book from a handful of synthetic PGNs (known moves, including
  a transposition); assert `total_games`, `support%`, conditional move counts, and that the
  transposition pooled counts under one EPD key. Assert the 0.1% prune drops a rare position.
- **`OpeningBook.lookup` (unit):** with a hand-built table — returns a legal book move when
  support ≥ threshold; returns `None` below threshold and for unknown positions; sampling with
  a seeded RNG is deterministic and only returns moves present in the table.
- **Integration (unit):** a `PolicyBot` with a stub book that returns a fixed move plays it;
  when the book returns `None`, it falls through to the model (mock/monkeypatch the model path).
- **Sanity (manual/script):** load a real band book; the start position's top book moves are
  sane (e4/d4/Nf3 dominate) and a deep/rare position falls through.

## Out of scope

Outcome-weighting of opening moves; blending book + model probabilities; Lichess-explorer
sourcing; books for elos outside 1000–1999 (clamped to the nearest band).

## Open items / risks

- **Build memory:** N=24 over 100k games yields many distinct deep positions before pruning;
  tally in a dict and prune at the end (openings concentrate, so the ≥0.1% set is modest), or
  prune incrementally if memory is tight. Resolve in the plan.
- **EP in EPD:** `board.epd()` emits the ep square conditionally (like FEN) — transposition
  merging remains correct; no action needed.
