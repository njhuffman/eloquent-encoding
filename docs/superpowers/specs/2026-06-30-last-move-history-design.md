# Last-move history (previous plies) — design

**Status:** validated in brainstorming; precedes the implementation plan.

## Goal

Give the model the **recent move history** (last up to 4 plies) so it can predict *reactive* moves —
recaptures, responses to fresh threats/checks, addressing newly-created tension — which a static
board cannot express. Mine generously (4 plies, cheap) once; make the number of plies used a
train/inference-time knob; keep the model usable when history is absent (puzzles / FEN-only).

## Why (value)

A static position hides **what just changed**. The biggest concrete win is **recaptures**: "opponent
just captured on e5" → recapture on e5, far more than "a piece is on e5." Also: respond-to-new-threat,
plan continuity (your own last move). This is *orthogonal* to Maia (position-only) — a lever to push
*past* ~50%, with precedent from AlphaZero/Lc0 (8 history planes). Expected gain is **concentrated on
reactive positions**; overall move-match may move modestly.

Key insight: from/to alone is **not** enough — whether the last move was a **capture (and of what)**
is NOT recoverable from the current board (the captured piece leaves no trace), and it is the
recapture trigger. The *moved* piece type IS recoverable (it sits on `to`) for the last ply, so we do
NOT store it.

## Two implementation areas

### A. Mining schema additions (`dataset_generation`)

For each emitted sample, record the **last H=4 plies** that led to `packed_pre` (ply-1 = most recent =
opponent's last move; ply-2 = the player's previous move; alternating). Three new HDF5 datasets,
each shape `(N, 4)` `int8`:
- `hist_from` — from-square 0–63, **sentinel 255 = absent ply**.
- `hist_to` — to-square 0–63, sentinel 255.
- `hist_cap` — captured piece type: **0 = no capture, 1=P 2=N 3=B 4=R 5=Q** (king never captured); for an absent ply, `hist_from==255` marks it (cap irrelevant).

Storage: 3×4×1 = **12 bytes/sample → ~+0.77 GB on 64M (~+20%)**. Cheap.

Mining is straightforward in the python-chess mainline replay (`candidate_collect`/`builder`): keep a
rolling buffer of the last 4 `(from, to, captured_type)`; `captured_type` from `board.is_capture(move)`
+ `board.piece_at(move.to_square)` before pushing. With `skip_opening_plies=4`, sampled positions
almost always have all 4 prior plies present (so absent-plies in *training* come mainly from dropout,
not game start — reinforcing the dropout design below).

Edge cases: **en passant** — captured pawn isn't on `to`; record `hist_cap=1` (pawn) attached to the
`to` square (minor, documented imprecision). **Castling** — recorded as the king's from/to (rook move
not separately tracked). **Promotion** in a prior ply — doesn't affect from/to/cap.

### B. Encoder marker design (`style_policy`)

Inject in the **encoder as additive square-markers** (spatial alignment beats head-injection for a
"which square just changed" signal). Flag-gated, zero-init (no-op until trained), mirroring the
castling/ep mechanism already merged.

Flags (architecture):
- `use_last_move: bool` (default **False**) — gates the whole feature.
- `n_history_ply: int` (default 4 when enabled) — number of per-ply marker sets the model has.

Parameters when enabled (all zero-initialized):
- `from_emb[i]`, `to_emb[i]` — learned d-vectors per ply `i` (recency-specific, so the model knows which change is freshest), `i = 0..n_history_ply-1`.
- `cap_emb` — `nn.Embedding(6, d)` over captured-type (0=none…5=Q), shared across plies, zero-init.

`encode()` when `use_last_move`: for each **present, non-dropped** ply `i`,
`tok[:, hist_from[i]] += from_emb[i]` and `tok[:, hist_to[i]] += to_emb[i] + cap_emb(hist_cap[i])`
(scatter-add; absent plies `hist_from==255` skipped; overlapping squares sum). So "a rook was just
taken on e5 (ply-1)" is a distinct, strong cue. The transformer then propagates it via attention.

Training knob — **history-horizon dropout** (NOT independent per-ply):
- `last_move_dropout: float` (e.g., 0.25) — per sample, with this probability truncate the history to
  its **K most-recent plies** (drop the older suffix), K drawn uniformly from `{0 .. available−1}`;
  otherwise keep the full available history.
- Rationale: history is **prefix-available** — you can never know ply-3 without ply-2 — so missingness
  is always a contiguous *older* suffix, never a gap (dropping ply-2 ⇒ drop ply-3 and ply-4). Independent
  per-ply dropout would train impossible configurations that never occur at inference.
- This is the missing-data mechanism (K=0 ⇒ history-less puzzles/FEN; game-start naturally yields short
  horizons), doubles as regularization, AND makes the training horizon distribution **identical to the
  inference K-sweep** (feed the K most-recent plies) — so train/inference align exactly.

## Variable-K experiment (the payoff of mining 4)

Train **one** model with `n_history_ply=4` + per-ply dropout, then sweep K **at inference** (feed only
the first K plies, mark the rest absent) → the diminishing-returns curve with **no retraining**.
Measure move-match overall AND on a **reactive subset** (positions where ply-1 was a capture/check —
where history should help most), vs the no-history `multiband_64M` baseline. Clean separate-K
trainings are a follow-up if the inference sweep is promising.

## Bundle with the next mine

History needs a re-mine anyway, so generate one combined next dataset: **last-4-ply history +
extended bands (add 2000–2200 strata) + WDL labels**. Train-time flags then select what each run uses
(bands present in data; `use_last_move`/`n_history_ply`; the castling/ep/cls flags already on). Make
the history columns **optional in `PackedMoveDataset`** (like `result`/`opp_elo`) so older datasets
still load; when `use_last_move=True`, training asserts the columns exist, and inference can supply
all-absent for history-less positions.

## Backward compatibility

`use_last_move` defaults False → existing checkpoints/configs/datasets unaffected (no new params, no
new columns required). New checkpoints store the flags in their architecture and rebuild correctly.

## Out of scope

Moved-piece-type (redundant for ply-1, low value beyond); full capture-train reconstruction for K>1;
exact ep captured-square; head-side injection; ONNX export of these features; the separate `use_cls`
band-head export/eval plumbing already noted elsewhere.
