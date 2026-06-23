# WDL value head (human-realized) — design

**Status:** approved design; precedes the implementation plan.

## Goal

Add a human-realized **WDL value head** — P(win / draw / loss | board, mover_elo) — to
the existing two-stage pointer policy, trained **jointly** on the shared encoder (not on a
frozen encoder). Validate at 16M positions, then scale to 64M. The value reflects *human*
outcomes (the realized game result), not engine evaluation.

Motivation: the bare imitation policy has no notion of winning and blunders in sparse
endgames. A jointly-trained value head forces the shared representation to encode
"who is winning," which should sharpen the policy and is the prerequisite for later
value-based engines (expectimax, trap-setting, KL-regularized tilt).

## Current state (verified)

- Encoder (`style_policy/board_encoder.py`) returns `(cls, square_tokens)`. The CLS output
  is **currently unused** — every head does `_, squares = self.encode(...)` and reads only
  the 64 square tokens. CLS is seeded by the side-to-move embedding and feeds turn info into
  the square tokens via attention, but its output vector is consumed by nothing. The value
  head is its first consumer.
- The packed training format (`/mnt/eloquence_bulk/databases/j3_training_*.h5`, read by
  `style_policy/dataset.py`) has columns:
  `packed_pre, packed_post, from_legal_u64, to_legal_u64, from_sq, to_sq, promotion, elo_to_move`.
  There is **no game result and no fen**. `packed_post` exists but training does not read it.
- The game `Result` header and both players' elos are parsed at collection
  (`dataset_generation/candidate_collect.py`) and then discarded.
- Source PGNs (`lichess_db_standard_rated_2025-01/02_tc_600_0.pgn.zst`) and all existing
  packed datasets are present at `/mnt/eloquence_bulk/databases/`.

## Decisions

1. **Joint training** on a shared encoder (not frozen-encoder probe).
2. **Validate at 16M first**, then regenerate + retrain at 64M.
3. **Store both elos** in the regenerated data; the **v1 value head conditions on the
   mover's elo only** (consistent with the policy). Opponent-elo conditioning is deferred
   and needs no further regen.
4. **Label = realized game outcome from the side-to-move's perspective** at each position
   (3-class W/D/L), taken from the PGN `Result` header.
5. **Value head input = the encoder CLS token** + a value-head-specific mover-elo embedding.
6. **Joint loss** `from_ce + to_ce + λ · wdl_ce`, starting `λ = 1.0`.

## Component 1 — Data regeneration

Regenerate the packed training dataset with two new per-row columns, reusing the **current**
codecs (`style_policy/board_encode.py`, `style_policy/packed_codec.py`) and the current
collection (`dataset_generation/candidate_collect.py`) — no revival of archived jepa3 build
code.

New columns:
- `result` (int8): outcome **from the side-to-move's perspective** at that position.
  Encoding: `loss = 0, draw = 1, win = 2`. Derived from the PGN `Result` header
  (`1-0`/`0-1`/`1/2-1/2`), inverted when Black is to move. Games with `Result = *`
  (unterminated) are dropped (already rare; the recipe filters short games).
- `opp_elo` (int16): the other player's elo (stored for future opponent-elo conditioning;
  unused by v1).

Schema notes:
- Keep the columns training reads (`packed_pre, from_legal_u64, to_legal_u64, from_sq,
  to_sq, promotion, elo_to_move`) plus the two new ones.
- **Drop `packed_post`** (YAGNI — training never reads it; the post-move board is
  reconstructable from `packed_pre` + the move if a future ΔV value-delta needs it).
- `style_policy/dataset.py` gains `result` in its returned batch dict (and `opp_elo`
  available but unused).

Builds: a 16M dataset first, then 64M, from the 2025-01/02 sources, matching the existing
recipe's filters (time control, skip-opening-plies, min-plies, single-legal-move exclusion).
The exact build entry point (extend `dataset_generation/builder.py` to emit the packed
schema directly vs. a thin fen→packed converter) is a plan-time decision; both reuse the
current codecs and add the two columns.

## Component 2 — Value head

`style_policy/value_head.py`: `WDLHead(nn.Module)`
- Inputs: `cls` `(B, d_model)`, optional `elo_idx` `(B,)`.
- Own elo embedding `nn.Embedding(n_elo_buckets + 1, elo_dim)` (mirrors `FromHead`/`ToHead`;
  index `n_elo_buckets` = unknown-elo, used when `elo_idx is None`).
- `score = MLP(d_model + elo_dim → head_hidden → 3)` returning raw `(B, 3)` WDL logits
  (order: loss, draw, win — matching the `result` encoding).

`BasePolicy` (`style_policy/model.py`):
- Construct a `WDLHead` in `from_config` (using existing `d_model`, `head_hidden`, `elo_dim`,
  `n_elo_buckets`).
- Add `forward_value(packed_pre, *, elo_idx=None)` returning WDL logits from the CLS token
  (the encode step already produces `cls`; stop discarding it).
- Extend `forward_policy` (the encode-once path) to also return value logits, so a training
  step encodes once and produces from/to/value together.

## Component 3 — Loss & training

- `style_policy/loss.py`: add `wdl_ce(value_logits, result)` = standard 3-class cross-entropy
  (mean over the batch) against the realized `result` label.
- `training_loop.py`: total loss `from_ce + to_ce + λ · wdl_ce`, `λ` configurable
  (default `1.0`). Log the three terms separately (W&B) plus a value metric (WDL accuracy /
  log-loss). NaN-guard unchanged.
- Same recipe as the base model: bf16 AMP, warmup + cosine, peak lr 2e-4, gradient
  accumulation, full retrain **from scratch** (existing checkpoints lack the value head).
- Model config: add `value_loss_weight` (λ) to the YAML; reuse `head_hidden`/`elo_dim`.

## Component 4 — Validation & success criteria (16M gate)

Before scaling to 64M, on the held-out validation set:
1. **WDL is real (not just base rate):** value log-loss / calibration beats an elo-only prior
   baseline (predicting the marginal W/D/L rate per elo bucket). I.e., the head learned
   position-dependent value. Report a calibration curve (predicted P(win) vs realized).
2. **Policy does not regress:** full-move top-1 (joint from+to) on the val set is
   neutral-or-better vs. the policy-only 16M baseline (`base_16M`). This is the core
   "does the aux head help or hurt the policy?" check.
3. **Stretch diagnostic (not a gate):** blunder rate / value-drop on a small held-out
   mate-in-one set — the original motivation. Informative for whether outcome-awareness
   reduces endgame blunders.

Go: criteria 1 + 2 pass → regenerate 64M + full joint retrain. No-go: value head hurts the
policy or fails to beat the prior → reconsider λ, head input (CLS vs mean-pooled squares),
or whether joint training is the right coupling.

## Out of scope (future work)

Opponent-elo conditioning; the ΔV / blunder-detector; any search or value-based engine
(expectimax, trap-setting, KL-tilt); using `packed_post`; adversarial/contrastive realism.
These are captured separately and build on this value head.

## Risks / open items

- **Label sparsity near game start vs. end:** a winning position late in a lost game still
  carries the game's final result — value is the *realized human outcome*, which is the
  intended (noisy but human) signal, not an oracle. Accept as designed.
- **λ balance:** value term could dominate or starve the policy; `λ` is the first knob to
  tune if criterion 2 fails.
- **Build wiring** (extend builder vs. converter) resolved in the plan; both reuse current
  codecs.
