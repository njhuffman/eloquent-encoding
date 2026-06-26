# Stockfish eval sidecar — design

**Status:** approved design; precedes the implementation plan.

## Goal

Score dataset positions with Stockfish 17.1 and write the results to a **sidecar HDF5** (the canonical
training/validation h5 is never modified), for **probing/testing** the model's value head and policy —
e.g. comparing the human-WDL value head against an objective engine eval. Two evals per position:

- **Static NNUE** (no search) — the engine's positional eval, tactically blind.
- **Depth-8 search** — eval after resolving immediate tactics — plus Stockfish's own W/D/L.

Not a training target (decided earlier); a derived analysis artifact, hence a sidecar.

## Context (settled earlier)

- Stockfish **17.1** installed at `/usr/games/stockfish` (apt `stockfish`; reinstall if the container is
  recreated). `python-chess` 1.11.2 with `chess.engine`.
- Benchmark: **depth 8 ≈ 6.5 ms/pos/core**; node limits are far heavier (avoid). First run = the held-out
  validation h5 (`/mnt/eloquence_bulk/databases/wdl_validation_1M.h5`, ~10k rows) — seconds at depth 8.
- Positions live as packed boards; `style_policy.board_encode.packed_to_board` reconstructs a
  `chess.Board` (move counters absent — irrelevant for eval; castling/ep preserved).
- The dataset's `result` column is **STM-relative**; all sidecar evals match that perspective.

## CLI — `scripts/eval_stockfish.py` (`python -m scripts.eval_stockfish`)

Args: `--h5` (default the validation file) · `--out` (default: source path with `.sf_eval.h5` suffix) ·
`--depth 8` · `--workers 8` · `--shard-size 5000` · `--sample N` + `--seed 0` (optional row subset) ·
`--stockfish /usr/games/stockfish` · `--hash 32`.

Selected rows = all rows, or a seeded random sample of size `N`. Reads `packed_pre` for those rows →
`packed_to_board` → `board.fen()`.

## Per-position eval (one engine set per worker)

Each worker owns: a `chess.engine.SimpleEngine` (configured `Threads=1, Hash=<hash>, UCI_ShowWDL=true`)
for the search, plus a tiny **raw `subprocess.Popen` Stockfish** for the static `eval` command
(python-chess doesn't expose `eval`). Engines/processes are cleaned up via `atexit` in the worker.

- **Searched**: `engine.analyse(board, Limit(depth=depth))` →
  - `info["score"].pov(board.turn)` → STM-relative `PovScore`; → `(cp, mate)` via `score_to_cp_mate`.
  - `info["wdl"].pov(board.turn)` → STM W/D/L permille (engine-reported, since `UCI_ShowWDL=true`).
- **Static NNUE**: write `position fen <fen>\neval\n` to the raw process, read until the
  `NNUE evaluation <±x.xx> (white side)` line, parse pawns→cp (`parse_static_eval`), flip to STM
  (`cp_static_stm = white_cp if board.turn==WHITE else -white_cp`).
- Terminal positions (shouldn't occur — training positions have a legal move): guard with
  `is_game_over()` → store sentinels (cp 0, mate 0, wdl 0,0,0) and mark done without calling the engine.

## Sidecar schema (HDF5, length = #selected rows; STM-relative)

- `row_index` int64 — the source-h5 row each eval maps to (joins back; sampling-proof).
- `sf_static_cp` int16 — static NNUE eval, clamped to ±32000.
- `sf_cp` int16 — depth-8 searched eval, clamped to ±32000 (on a mate, set to the clamp sentinel).
- `sf_mate` int8 — signed plies to mate (+ = STM mates, − = STM gets mated), 0 = no forced mate.
- `sf_wdl` int16 shape (N,3) — W/D/L permille, order **[loss, draw, win]** STM (matches the model's WDL
  head ordering loss=0/draw=1/win=2).
- `done` bool (N,) — per-row completion flag (resumability).
- Attrs: `source_h5`, `source_n_rows`, `depth`, `stockfish_version`, `hash_mb`, `sample_n`, `seed`,
  `perspective="STM"`, `wdl_order="loss,draw,win"`, `cp_clamp=32000`.

## Resumability

Open-or-create the sidecar sized to N, datasets pre-filled with sentinels and `done=False`. Process
selected rows in shards of `--shard-size`; for each shard, eval the not-yet-`done` rows, write their
slices, set `done[...]=True`, and flush. Re-running the same command **skips rows already `done`** (so a
killed multi-hour run resumes). Validate on open that the sidecar's `source_h5`/`source_n_rows`/`sample`
match the request (else refuse, to avoid mismatched alignment).

## Parallelism

`multiprocessing.Pool(workers)` (default **8**, to leave cores free) with an initializer that opens the
per-worker engine set. Map row chunks via `imap_unordered`; the main process writes results into the
sidecar as they stream and marks `done`. Memory ≈ 8 workers × 2 SF processes × ~165 MB ≈ a few GB — fine.

## Components / files

- `dataset_generation/stockfish_eval.py` — the reusable pieces (pure + engine wrapper):
  - `parse_static_eval(text: str) -> int` — pawns→cp from the `NNUE evaluation` line (handles ±, "side").
  - `score_to_cp_mate(pov_score) -> tuple[int, int]` — `(cp, mate_plies)`; mate → cp clamp sentinel.
  - `clamp_cp(cp: int) -> int` — clamp to ±32000.
  - `StaticEvalEngine` — thin wrapper around the raw `eval` subprocess (`eval_cp(fen) -> int`).
  - `eval_position(simple_engine, static_engine, board, depth) -> dict` — returns the per-row record.
- `scripts/eval_stockfish.py` — CLI: arg parsing, row selection, sidecar open/resume, the worker pool,
  shard writing.
- Tests `tests/dataset_generation/test_stockfish_eval.py`:
  - Pure: `parse_static_eval` (e.g. `"NNUE evaluation  +0.49 (white side)"` → 49; negative; large),
    `score_to_cp_mate` (cp passthrough+clamp; mate +N / −N → sentinel cp + signed plies),
    `clamp_cp` bounds.
  - Resume logic: a sidecar pre-seeded with some `done=True` rows → the selector yields only the
    not-done rows (no engine needed).
  - Stockfish-gated integration (skip if `/usr/games/stockfish` absent): eval ~5 fixed FENs → columns
    populated, `sf_cp`/`sf_static_cp` within int16, `sf_wdl` rows sum ≈ 1000, a known mate FEN → nonzero
    `sf_mate`.

## First run

`python -m scripts.eval_stockfish` (defaults) → writes
`/mnt/eloquence_bulk/databases/wdl_validation_1M.sf_eval.h5` for the ~10k validation rows at depth 8 on
8 workers (seconds). Spot-check a few rows vs the board (sane signs/magnitudes).

## Out of scope

Wiring SF eval into training or the web; the full-16M run (the script scales to any file/`--sample`
later); deeper analyses comparing it to the model (separate, downstream).

## Risks

- **python-chess `wdl` availability**: if `info["wdl"]` is absent for a position, store `[0,0,0]` and
  continue (don't crash the batch). Confirm `UCI_ShowWDL` populates it in the integration test.
- **Static-eval parsing drift**: pin the parse to the `NNUE evaluation … (white side)` line; the
  integration test guards the format against SF version changes.
- **Engine process leaks**: `atexit` quit/terminate in workers; the script also `pkill`s stray engines on
  exit. (Keep worker count ≤8 so the box stays usable.)
- **Alignment**: `row_index` + the source-match check on resume prevent a sidecar being joined to the
  wrong file/order.
