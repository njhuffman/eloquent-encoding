# Web opening-book integration — design

**Status:** approved design; precedes the implementation plan.

## Goal

Make the deployed GitHub Pages bot use the per-elo-band opening book (already built in Python at
`/mnt/eloquence_bulk/databases/opening_book/`), so it opens with human-frequency moves and hands
off to the model in the early middlegame — fixing the deployed bot's out-of-distribution move-1
behavior. The book is consulted before the ONNX model; on a miss it falls through to the model.

## The key-parity decision (the crux)

The book is keyed by python-chess `board.epd()` = `<placement> <side> <castling> <ep>`. The browser
(chess.js) must reproduce that key **byte-for-byte** or lookups silently miss. Fields 1–3 are
identical across the libraries (standard FEN). The only divergence is **en passant**: python-chess
writes the ep square **only when an ep capture is actually legal** (else `-`); chess.js writes it
after any double push. We **keep the existing epd keys (no re-key, no rebuild)** and reproduce them
in TS with an ep normalization:

- `epdKey(board)` = `${placement} ${turn} ${castling} ${ep}` where the first three come from
  `chess.js` `board.fen()` split, and **`ep` is computed, not taken from the raw FEN**: scan
  `board.moves({ verbose: true })`; if any move's flags include `e` (en passant), `ep` = that
  move's `to` square (the ep target); otherwise `ep = "-"`. This matches python-chess `epd()`
  exactly, EP included where it matters.

A **parity fixture** verifies it: the build emits `{epd, fen}` pairs for ~20 positions *including
ep-able ones*; a TS test reconstructs each from `fen`, computes `epdKey`, and asserts it equals the
stored `epd`.

## Components

### 1. Ship the book + parity fixture (build side)
- Copy the 10 band JSONs to `web/public/opening_book/band_<band>.json` (committed; ~1.8MB total,
  cached after first load — fine for Pages). A small script/step does the copy.
- `scripts/gen_epd_fixtures.py`: emit `web/src/inference/__fixtures__/epd_cases.json` =
  `[{epd, fen}]` for a fixed set of positions including ep-able ones (the parity oracle).

### 2. `web/src/inference/openingBook.ts`
- `epdKey(board: Chess): string` — as defined above (ep-normalized).
- `eloToBand(elo: number): number` — `clamp(floor(elo/100)*100, 1000, 1900)` (mirrors Python).
- `class OpeningBook` (one band): `{ totalGames, positions: {epd: {n, moves: {uci:count}}} }`;
  `lookup(board, threshold, rand: () => number): {from,to,promotion?} | null` — if `epdKey(board)`
  present and `n/totalGames ≥ threshold`, sample a **legal** move ∝ conditional counts using
  `rand`; else `null`. Mirrors the Python `OpeningBook.lookup`.
- `class OpeningBookSet` — lazy per-band fetch + cache: `forElo(elo): Promise<OpeningBook | null>`
  fetches `${BASE_URL}opening_book/band_${eloToBand(elo)}.json` once per band, caches it, returns
  `null` if the file is absent (book simply disabled).

### 3. Integration (`web/src/components/BoardPanel.tsx` + `web/src/useEngine.ts`)
- Load an `OpeningBookSet` alongside the engine (in `useEngine`).
- In `botMove`, **before** `engine.chooseMove`: `const bk = await books.forElo(elo); const mv =
  bk?.lookup(game, BOOK_THRESHOLD, Math.random); if (mv) { apply mv } else { engine.chooseMove(...) }`.
  `BOOK_THRESHOLD = 0.01` (1%, matches the Python default). Move-application + last-move/thinking
  state unchanged. Sampling uses `Math.random` (each game varies; tests inject a seeded `rand`).
- The "what the model is thinking" panel is unchanged: it keeps showing the model's top moves for
  the current position. The book only changes which move the *bot plays* — no panel special-casing.

### 4. Tests
- **Parity (the critical one):** for every `epd_cases.json` entry, `epdKey(new Chess(fen))` ===
  `epd` — proving the key (ep included) matches python-chess byte-for-byte.
- **`epdKey` unit:** an ep-able position yields the ep square; a double-push-with-no-legal-capture
  yields `-`.
- **`OpeningBook.lookup`:** threshold gate (none below threshold / unknown), ∝-counts sampling
  (seeded determinism + distribution), legal-filter.
- **`OpeningBookSet.forElo`:** maps elo→band, returns `null` on a missing file.
- **Integration:** `botMove` plays the book move when in-book; falls through to the model on a miss
  (stub the engine).

## Data flow

```
botMove(board, elo):
  book = await books.forElo(elo)                       # lazy fetch+cache band file
  mv = book?.lookup(board, 0.01, Math.random)          # epdKey -> support>=1%? sample ∝ counts
  if mv: play mv                                        # in-book human-frequency move
  else:  play engine.chooseMove(board, elo, opts)      # out of book -> model (unchanged)
```

## Out of scope

Showing book move stats in the thinking panel; a book on/off or threshold UI control (fixed 1%);
re-keying the book; books for elos outside 1000–1999 (clamped to nearest band); shipping a smaller
pruned book (all 10 bands at ~1.8MB is fine).

## Risks

- **Key parity (ep):** mitigated by the computed ep-normalization + the parity fixture (the gating
  test). If a future chess.js change alters move flags, the fixture catches it.
- **First-load size:** +1.8MB book on top of the ~7MB model + ORT WASM. Acceptable; lazy per-band
  fetch means only the played band loads.
