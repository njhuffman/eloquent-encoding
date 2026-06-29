# Band-specialized heads on a frozen encoder — design

**Status:** validated in brainstorming; precedes the implementation plan.

## Goal

Test whether **per-band specialized policy heads on a frozen, elo-agnostic encoder** sharpen the
elo "diagonal" (band-X conditioning best predicts band-X play) versus the current single shared
head conditioned by a learned elo embedding. Stretch goal: also widen elo→strength steering.

## Motivation

`diagonal_check` shows the shared elo-conditioning is a weak lever: the diagonal is the column-best
in only 2/10 bands with edges ~+0.5% (inside noise), and this is **unchanged by 3.3× model scale**
(7M vs 22.6M) — so it's structural, not capacity/data. A per-sample "elo-discrimination" loss was
rejected: a single sample is one draw from `P(move|pos,elo)`, so forcing "the true elo played this,
others wouldn't" injects bias on the (majority) shared-move positions.

A **bias-free** alternative (adapted from Maia-1's per-rating models): train each band's head on
**only that band's data** with plain cross-entropy. Each head learns `P(move|pos,band)` for its band
with full, dedicated capacity — no thin elo embedding, no shared-weight dilution, no fabricated
exclusivity. The encoder stays elo-agnostic (frozen, shared) — the user's hard constraint
([[elo-agnostic-encoder]]).

## Key question the experiment answers

A frozen elo-agnostic encoder emits the **same features for every band**, so per-band heads can only
sharpen the diagonal to the extent those features already contain band-discriminating information.
The quick test is decisive either way:
- **Specialized head sharper** → the signal is in the features; specialization unlocks it (and the
  later joint-trained version will do better still).
- **Specialized head ≈ shared** → the features lack band signal; a sharper diagonal *requires* the
  encoder to co-adapt (joint shared-encoder + per-band-heads), so frozen alone won't suffice.

## Approach

1. Load a trained checkpoint (default `base_64M` — strongest policy, most data → richest features;
   directly comparable to its existing diagonal baseline). Freeze the encoder (eval + `no_grad`).
2. Train an **unconditioned** `FromHead`/`ToHead` (`elo_dim=0`) on a single band's rows (CE on
   from/to with legality masking), reading frozen encoder features.
3. Evaluate the band head's move-match across **all** bands (its diagonal row) and compare, on the
   same eval rows, to the shared head conditioned at each band.

Quick test = one band (1900 — the tail with the largest current spread, most likely to reveal a
gain). If promising, scale to all 10 bands for a full specialized-vs-shared diagonal.

## Components (reused)

- `style_policy/model.py`: `BasePolicy.from_config`, `.encode` → `(cls, squares)`.
- `style_policy/policy_heads.py`: `FromHead`/`ToHead` (support `elo_dim=0`).
- `style_policy/dataset.py`: `PackedMoveDataset` (+ new band filter).
- `style_policy/loss.py`: `masked_square_ce`, `top1_legal`, `joint_top1`.
- `style_policy/legal_mask.py`: `u64_to_mask`; `model_spec.elo_to_bucket`.
- `scripts/diagonal_check.py`: the move-match metric to mirror.

## New components

- `band_indices` (in `dataset.py`): rows with `band_lo ≤ elo_to_move < band_hi`.
- `style_policy/band_head.py`: `BandHead` (from+to, `elo_dim=0`); `train_band_head(...)` (frozen
  encoder → train head → save); `eval_band_head_row(...)` (move-match across bands + shared-head
  baseline on the same rows).
- `scripts/train_band_head.py`, `scripts/eval_band_head.py`: thin CLIs.

## Decision criteria (band 1900 quick test)

Specialization is "promising" if, on the same eval rows:
- band-1900 head's move-match on band-1900 **> shared head @1900 on band-1900** (beats the shared
  head on its own band), AND
- the band-1900 head's row is **peaked at 1900** (argmax over bands) with a larger off-diagonal drop
  (1900 vs 1000) than the shared head shows.

## Out of scope (this experiment)

Joint shared-encoder + per-band-heads training (the "technically better" version — a follow-up if
the frozen test is promising); per-band value heads; smooth elo interpolation between heads;
deploying band heads to the web bot.

## Compute note

Needs the GPU; the in-flight `wdl_16M_big` run holds ~3.5/4GB (a concurrent load OOMs). Build +
unit-test now (CPU/tiny data); run the GPU experiment when training frees up (~4h), or CPU-cache a
subsample for an early peek.
