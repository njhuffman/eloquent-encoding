# Last-Move History Implementation Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. See spec 2026-06-30-last-move-history-design.md.

**Goal:** Mine the last 4 plies `(from,to,captured_type)` per sample (bundled with band-extension to 2200 + WDL), and add flag-gated encoder markers + history-horizon dropout so the model uses recent move history, with a train/inference K knob and graceful absence.

## Global Constraints

- Everything flag-gated, **default off**: `use_last_move` (+ `n_history_ply`, `last_move_dropout`). Existing checkpoints/datasets/configs unaffected. History columns OPTIONAL in the loader.
- Markers zero-init (no-op until trained), additive scatter-add, mirroring the merged castling/ep mechanism.
- History is **prefix-available**: dropout truncates the OLDEST plies (drop ply-2 ⇒ drop 3,4), K∈{0..available}.
- Captured-type encoding: `0=none, 1=P, 2=N, 3=B, 4=R, 5=Q` (= python-chess piece_type for P..Q).
- **Absent-ply sentinel = `-1`** (columns are int8, so 255 is invalid — D1 settled on -1). Present ⇔ `hist_from >= 0`. **CRITICAL: never index a token with a negative square** — mask absent plies out before any `tok[hf]` indexing (a -1 index would silently hit square 63).
- Tests hermetic, run in container: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && PYTHONPATH=. python -m pytest <p> -q'`. No `tests/style_policy/__init__.py`.

---

## Stream 1 — Data generation

### Task D1: history columns in the HDF5 writer

**Files:** Modify `dataset_generation/hdf5_io.py`; Test `tests/dataset_generation/test_hist_writer.py`.

- Add three `(N,4) int8` datasets to `PackedBatchWriter`: `hist_from`, `hist_to`, `hist_cap`. Add a `_HIST = ("hist_from","hist_to","hist_cap")` group; create them in `__init__` with `shape=(0,4), maxshape=(None,4), dtype=np.int8, chunks=(CHUNK,4)`; in `flush`, handle them like `packed_pre` (2D resize to `(o+m,4)`); `append_row` gains `hist_from, hist_to, hist_cap` params (each a length-4 sequence) appended to the buffers. Keep all existing columns/behavior.
- **Test:** write 3 rows with known hist arrays (incl. a row of all-absent (-1)), reopen, assert shapes `(3,4)` and values round-trip.
- Commit: `feat(datagen): hist_from/to/cap columns in PackedBatchWriter`.

### Task D2: track last-4-ply history in mining

**Files:** Modify `dataset_generation/candidate_collect.py`, `dataset_generation/builder.py`; Test `tests/dataset_generation/test_hist_collect.py`.

- In `collect_candidate_positions`, maintain `recent = collections.deque(maxlen=4)` of `(from_sq, to_sq, cap)` for pushed moves (most-recent first). For each `move`, BEFORE `board.push(move)` compute `cap`: `0` if not `board.is_capture(move)` else `chess.PAWN(=1)` if `board.is_en_passant(move)` else `board.piece_at(move.to_square).piece_type` (1..5; king never captured). At emit time, build `hist = [(f,t,c) for the up-to-4 entries already in recent]` newest-first, padded with `(-1,-1,0)` to length 4. Append `move`'s `(from,to,cap)` to `recent` AFTER emit/push. Add `hist` to each emitted row tuple → `(ply, stm, elo_tm, opp_elo, result, move, hist)`.
- In `builder.py`: unpack the extra `hist` element and pass `hist_from=[h[0] for h in hist]`, `hist_to=...`, `hist_cap=...` to `writer.append_row(...)`.
- **Test:** construct a short `chess.pgn.Game` with a known capture (e.g. `1.e4 d5 2.exd5 ...`), `skip_opening_plies=0`, and assert the emitted row for the position after `2.exd5` has `hist[0]=(from=e4-ish, to=d5, cap=1)` (the pawn capture) and earlier/absent plies padded with -1; also a position with fewer than 4 prior plies pads correctly.
- Commit: `feat(datagen): mine last-4-ply (from,to,captured_type) history`.

### Task D3: combined recipe (history + band-extension + WDL)

**Files:** Create `dataset_generation/wdl_history_64M.yaml`.

- Clone `wdl_training_64M.yaml` (Jan+Feb, WDL schema — history now emitted automatically by D1/D2). ADD strata `elo_min/max` `2000–2099` and `2100–2199` to BOTH source_plans, `samples_per_game: 8`, `take_games: 400000` (the builder takes `min(take_games, available)`; 2000–2199 are scarcer, so they fill with whatever's present — log the realized counts). Do NOT add 2200+ (too sparse). `name: wdl_history_64M`.
- No test (config). Note in a comment that this is the next overnight gen.

---

## Stream 2 — Architecture

### Task A1: optional history columns in PackedMoveDataset

**Files:** Modify `style_policy/dataset.py`; Test add to `tests/style_policy/test_band_filter.py` or new `test_hist_dataset.py`; extend `tests/style_policy/synth_h5.py` to optionally write hist columns.

- `__getitem__`: if the file has `hist_from`, load `hist_from/hist_to/hist_cap` for the row as `int64` tensors (shape `(4,)`); else return all-absent (`hist_from=full(4,-1)`, `hist_to=full(4,-1)`, `hist_cap=zeros(4)`). Add to the returned dict; `collate` already stacks all keys → batched `(B,4)`.
- **Test:** synth h5 WITH hist → dataset returns the stored values; synth h5 WITHOUT hist → returns all -1/0 absent.
- Commit: `feat(dataset): optional last-move history columns (absent-by-default)`.

### Task A2: encoder history markers (flag-gated)

**Files:** Modify `style_policy/board_encoder.py`, `style_policy/model.py` (`from_config`, `encode`, `forward_policy/from/to`), `style_policy/multiband_policy.py` (`from_config`, `encode`); Test `tests/style_policy/test_last_move.py`.

- `BoardEncoder.__init__(..., use_last_move=False, n_history_ply=4)`. When `use_last_move`: `self.from_emb = nn.Parameter(torch.zeros(n_history_ply, d_model))`, `self.to_emb = nn.Parameter(torch.zeros(n_history_ply, d_model))`, `self.cap_emb = nn.Embedding(6, d_model)` (zero-init). Store `n_history_ply`.
- `BoardEncoder.forward(self, board_tensor, hist=None)`: after building `tok` (and the existing castling/ep additions), if `self.use_last_move and hist is not None`: `hf, ht, hc = hist` (each `(B,4)` long). For `i in range(n_history_ply)`: `present = hf[:,i] >= 0` (mask absent; never index with a negative square); scatter-add `from_emb[i]` into `tok` at `hf[:,i]` for present rows, and `to_emb[i] + cap_emb(hc[:,i].clamp(0,5))` at `ht[:,i]` for present rows. (Vectorize with masked index_add per ply, or build a `(B,64,d)` additive tensor; either is fine — markers for absent rows must contribute zero.)
- `BasePolicy.from_config`/`MultiBandPolicy.from_config`: pass `use_last_move=bool(cfg.get("use_last_move", False))`, `n_history_ply=int(cfg.get("n_history_ply", 4))` to `BoardEncoder`.
- `BasePolicy.encode(self, packed_pre, hist=None)` → `self.encoder(board, hist=hist)`. `MultiBandPolicy.encode` same. `forward_policy/forward_from/forward_to` gain `hist=None` and pass it to `encode`.
- **Test:** (a) `use_last_move=False` → output identical to a no-hist encoder (passing hist ignored); (b) `use_last_move=True`, zero-init → output equals the no-hist case (no-op until trained); (c) after randomizing `from_emb/to_emb/cap_emb`, two hist inputs differing in one ply produce different `(cls,squares)`; absent (`-1`) plies contribute nothing.
- Commit: `feat(encoder): optional last-move history markers (use_last_move, default off)`.

### Task A3: history-horizon dropout + training wiring

**Files:** Modify `style_policy/multiband_train.py` (`_step`, `_routed_policy_loss` already takes cls — add hist), and add a helper `style_policy/history.py` with `horizon_dropout`; Test `tests/style_policy/test_history_dropout.py`.

- `style_policy/history.py`: `horizon_dropout(hist_from, hist_to, hist_cap, p, gen) -> (hf,ht,hc)`: per sample, with prob `p` draw `K ~ Uniform{0..available-1}` (available = count of `hist_from[row] >= 0`) and set plies `>=K` (the oldest) to absent (`from/to=-1, cap=0`); else unchanged. Newest-first ordering means "keep first K".
- `multiband_train._step`: read `hist_from/hist_to/hist_cap` from the batch, apply `horizon_dropout(..., p=stage.get("last_move_dropout",0.0))` (training only), pass `hist=(hf,ht,hc)` into `model.encode(packed, hist=...)`. (Update `_routed_policy_loss`/`_step` so the encode call carries hist; cls handling unchanged.)
- Read `use_last_move` etc. are arch flags (already in `from_config`); `last_move_dropout` is a stage/train knob.
- **Test:** `horizon_dropout` with `p=1.0` and a fixed gen truncates to a contiguous newest-prefix (older plies become -1, no gaps); with `p=0.0` is identity; K=0 occurs (full absence reachable).
- Commit: `feat(train): history-horizon dropout + wire hist through multiband encode`.

---

## Stream 3 — Train + eval (gated on Stream 1 gen)

### Task T1: config + generate + train

**Files:** Create `style_policy/model_configs/multiband_history_64M.yaml` (clone `multiband_64M.yaml`; `train_h5: .../wdl_history_64M.h5`, add `use_last_move: true`, `n_history_ply: 4` to architecture, `last_move_dropout: 0.25` to defaults; bands stay 1000–1900 OR extend to 2200 once data confirms — decide from D3 realized counts).

- [ ] Run the gen: `python -m dataset_generation build --recipe dataset_generation/wdl_history_64M.yaml --data-dir /mnt/eloquence_bulk/databases --output-dir /mnt/eloquence_bulk/databases` (overnight; capped cores). Confirm `hist_*` columns + row count + 2000–2199 band counts.
- [ ] Train: `python -m scripts.train_multiband --model multiband_history_64M --device cuda` (~overnight; W&B).

### Task T2: inference K-sweep eval

**Files:** Create `scripts/eval_history_ksweep.py`.

- Load the trained model; on `wdl_validation` (regenerate val WITH history, or reuse if mined), for K in 0..4 feed only the K most-recent plies (set plies `>=K` to -1) and measure full-move top-1 — overall AND on a **reactive subset** (rows where `hist_cap[0] != 0` i.e. ply-1 was a capture, and/or ply-1 gave check). Compare to the no-history `multiband_64M` baseline (K=0 ≈ baseline).
- [ ] Run + record the diminishing-returns curve + reactive-subset deltas to memory.

---

## Self-Review
- Coverage: writer (D1), mining (D2), recipe (D3), dataset (A1), encoder markers (A2), dropout+wiring (A3), config+gen+train (T1), K-sweep eval (T2).
- Backward-compat: all flags default off; hist columns optional; markers zero-init/additive — existing checkpoints/datasets/scripts unaffected.
- Prefix-truncation dropout matches inference K-sweep exactly.
- Note: the `use_cls` band-head export plumbing (separate prior follow-up) is also needed before rating a flagged multiband model — fold its fix into T1's train/export path when executing.
