# Parallel, training-ready data mining — design

**Status:** validated in brainstorming; precedes the plan.

## Goal

Make data generation use all cores instead of one (~8–10× wall-clock), and emit a
**training-ready** file (globally shuffled on disk → sequential reads, no per-epoch random access).
Target: the held `wdl_history_64M` gen drops from ~overnight to ~1–2 h and trains with a sequential
dataloader. No on-disk schema change; no builder-internals change.

## What already exists (do NOT rebuild)

- **Header prefilter** — `dataset_generation/stream.py::_iter_filtered_pgn_game_texts` +
  `pgn_prefilter.py::passes_header_prefilter` already regex-parse `WhiteElo`/`BlackElo`/`TimeControl`
  from headers and **skip movetext** (never `read_game`) for any game that can't match an unfilled
  stratum, and **early-exit** the stream once all of a plan's quotas are met
  (`all_strata_quotas_met`). So a worker restricted to one band already skips every other band at
  cheap regex cost. No pre-filter task is needed.

## Why current mining is slow

`dataset_generation` has **zero parallelism** — one process, ~1 of 22 cores. The dominant cost is
`chess.pgn.read_game` + mainline replay (`collect_candidate_positions`) on every *kept* game,
summed over **all bands** on a single core.

## Approach (3 parts)

### 1. Band-sharded independent workers (no builder change)

A driver (`scripts/mine_parallel.py`) splits the recipe's `(source_plan × stratum)` cells into G
shards and launches G `python -m dataset_generation build` processes (concurrency capped to the core
budget), each writing its own `{name}_shardNN.h5`.

**Sharding mechanism — `take_games` zeroing (preserves bit-identical sampling):** each shard's
sub-recipe is the *full* recipe with `take_games` set to **0** for every cell except the band(s)
assigned to that shard. This matters because `builder._rng_for_game` seeds each game's sampling on
`[master_seed, source_plan_index, stratum.stratum_seed, stratum_index, g]`. Keeping all strata in
their **original positions** (merely zeroed) preserves `source_plan_index`/`stratum_index`/
`stratum_seed`, so a shard samples its band **bit-identically** to the single-process run → the merged
union is an exact superset of (= byte-identical, as a set, to) today's output. A zeroed stratum has
`accepted=0 ≥ take_games=0`, so the prefilter treats it as already-full (never buffered) and
`_ensure_strata_quotas_met` does not flag it (`0 < 0` is false) — no builder edit required.

**Grouping / load balance:** keep each shard's non-zero cells within a **single source** (so a file
is decompressed only once per shard touching it). Partition each source's bands into groups balanced
by `take_games`, choosing G ≈ the core budget (≤14). Common bands fill fast and the worker
early-exits; scarce/late bands scan the whole file but parse little. Wall-clock ≈
`floor(decompress + regex-scan whole file)` + `parse(slowest single shard's bands)`, vs single-process
`floor + parse(all bands)` — speedup gated by the shared floor (regex-scan ≪ parse ⇒ good speedup).

### 2. Shuffle-merge (training-ready output)

Concatenating shards yields a **band-sorted** file (bad for sequential reads). The merge
(`dataset_generation/shuffle_merge.py`) instead does a **global, seeded shuffle**: total
`N = Σ shard rows`; draw one seeded permutation of `[0, N)`; for **each HDF5 dataset**
(`packed_pre`, `from_legal_u64`, `to_legal_u64`, `from_sq`, `to_sq`, `promotion`, `elo_to_move`,
`opp_elo`, `result`, `hist_from`, `hist_to`, `hist_cap`), concatenate across shards → apply the **same**
permutation → write. Processing one dataset at a time bounds peak RAM to ~2× the largest dataset
(`packed_pre`); total ~5 GB at 70M fits on this box. Seeded ⇒ reproducible. (External two-level
bucket-shuffle is the documented fallback if RAM-tight; not implemented — YAGNI.)

### 3. Sequential-read dataset path

With a pre-shuffled file + **single-epoch** training, one in-order pass = a full shuffle with **zero
random reads**. `PackedMoveDataset` currently subsamples via `rng.choice` (random indices → random
reads); add a **sequential mode** (take the first-N rows / iterate in order) selected by a config flag
(default off → existing random behavior preserved for un-shuffled files), and train with
`shuffle=False`. Frees dataloader CPU and is required for IID once the data is pre-shuffled on disk;
won't cut wall-clock while training stays compute-bound.

## Reproducibility

- `take_games`-zeroing preserves seed indices ⇒ each band sampled bit-identically to single-process.
- Band-per-shard: exact first-N-in-file-order semantics per band (the prefilter is order-preserving).
- Shuffle-merge: single seeded permutation.

## Caveats / risks

- **Duplicated decompression** (each source decompressed once per shard touching it) — bounded by
  grouping bands per shard; cheap vs parse.
- **Scarce high bands** (`2000–2199` at `take_games=400000`) may not exist in the month files;
  `_ensure_strata_quotas_met` **raises** if a quota is unmet — single-process has this same risk. The
  driver must surface a shard's non-zero failure loudly (not silently drop a band). Lowering those
  `take_games` or adding a "take up to available" builder option is a separate recipe/builder concern,
  out of scope here.
- RAM for in-RAM shuffle (~5 GB at 70M) — fine here; external fallback documented, not built.

## Out of scope

The actual parallel `wdl_history_64M` run (HELD); seekable-zstd to avoid duplicated decompression;
the external bucket-shuffle fallback; any on-disk schema change; "take up to available" quota
semantics.
