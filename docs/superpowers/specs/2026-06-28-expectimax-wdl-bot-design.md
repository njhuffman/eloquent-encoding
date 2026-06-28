# Expectimax (WDL) search bot — design

**Status:** approved design; precedes the implementation plan.

## Goal

A depth-limited **expectimax** bot that uses the model's policy to propose moves and model the
opponent, and the **WDL value head** to evaluate leaf states, playing the move with the best expected
WDL for the bot. Then rate it at several search depths vs Maia2 to see whether deeper search helps.

Built on **`wdl_16M`** (the only checkpoint with a trained value head) for both the policy (move
proposals + opponent model) and the leaf value. Maximizes expected **human-outcome** WDL — a solid
positional eval (Spearman ~0.85 vs Stockfish), tactically limited; deeper search adds explicit
opponent-reply lookahead on top of the head's implicit lookahead. That gap is what we're testing.

## Context (reused)

- `style_policy/play.py`: `Player.choose_move(board)->Move` seam; `play_match`.
- `style_policy/model.py`: `BasePolicy.encode`, `.from_head`, `.to_head`, `.value_head`.
- `style_policy/board_encode.py`: `board_to_packed`, `legal_from_u64`, `legal_to_u64`; `legal_mask.u64_to_mask`.
- The Maia2 rating tool: `style_policy/maia2_bot.py` (`load_maia2`, `Maia2Bot`), `style_policy/rating.py` (`mle_rating`), `scripts/rate_bot.py` (`bot_record_vs`).

## Components

### 1. `style_policy/search.py` — pure expectimax (no chess/torch)
```
expectimax(node, depth, expand, leaf_value, is_max_node) -> (value: float, best_move)
```
- `depth <= 0` → `(leaf_value(node), None)`.
- `children = expand(node)` returns `[(move, child, prob), …]`; if empty (terminal) → `(leaf_value(node), None)`.
- `is_max_node(node)` True (bot to move) → return the child with the **max** recursed value, and its move.
- else (chance / opponent to move) → `(Σ (probᵢ/Σprob)·value(childᵢ), None)` — probability-weighted
  expectation over the (renormalized top-K) children.
Generic over node/move types via the three callables → unit-testable on a toy tree.

### 2. `style_policy/search_bot.py` — `policy_topk` + `ExpectimaxBot(Player)`
- `policy_topk(model, board, elo_idx, k, device) -> [(move, prob)]`: encode once; `P(from)` via masked
  softmax of `from_head`; for each distinct legal `(from,to)` compute `P(from)·P(to|from)` (cache
  `to_head` per from-square); represent a promotion `(from,to)` by its **queen** move (the policy is
  from/to only); return the top-`k` `(move, prob)` descending.
- `ExpectimaxBot(Player)`: `__init__(self, checkpoint, elo, depth, *, width=4, device="cpu", seed=0,
  opening_book=None, book_threshold=0.01)` loads `wdl_16M`, caches `_elo_idx`.
  - `choose_move(board)` (under `torch.no_grad`):
    1. opening book (if set) → return its move on a hit.
    2. `depth == 0` → return `policy_topk(...)[0]` move (raw-policy argmax baseline; no value/search).
    3. else `bot_color = board.turn`; run `expectimax(board, depth, expand, leaf_value, is_max_node)`
       and return the best root move. Closures:
       - `expand(b)` = `[(mv, b.copy()+push(mv), prob) for mv, prob in policy_topk(model, b, _elo_idx, width, device)]`.
       - `is_max_node(b)` = `b.turn == bot_color`.
       - `leaf_value(b)`: terminal → exact bot-POV escore (`1.0` win / `0.5` draw / `0.0` loss by
         `b.outcome().winner` vs `bot_color`); else value-head **escore** (`P(win)+0.5 P(draw)`,
         STM, conditioned on `_elo_idx`) flipped to bot POV (`escore if b.turn==bot_color else 1-escore`).
  - Deterministic (top-K, no sampling); `seed` only seeds the opening-book RNG.
- Opponent chance-nodes use the **same model at the bot's elo** (single-elo assumption, documented).

### 3. `scripts/rate_search.py` — depth sweep vs Maia2
- Args: `--checkpoint` (default `style_policy_checkpoints/wdl_16M/wdl_16M_stage_1.pt`), `--elo 1500`,
  `--width 4`, `--depths 0 1 2 3`, `--levels 1100 1300 1500 1700 1900`, `--games-per-level 25`,
  `--device cuda`, `--book/--no-book` (default on, book for the bot's elo), `--max-plies 300`,
  `--seed 0`, `--out` (optional JSON).
- `load_maia2` once. For each depth: build `ExpectimaxBot` at that depth; for each Maia level run
  `bot_record_vs` (color-balanced) → W/D/L → score → `mle_rating`. Print a `depth → rating ±CI`
  table with per-level scores; stream per depth (so 0/1/2 print before the slow depth 3).

## Data flow

```
choose_move(board):                      # bot to move
  if depth==0: return policy_topk(board)[0].move
  v, move = expectimax(board, depth,
              expand     = top-K policy children (push move),
              leaf_value = bot-POV WDL escore (terminal: exact),
              is_max_node= board.turn == bot_color)
  return move
rate_search: for d in depths: rate ExpectimaxBot(depth=d) vs Maia2 levels -> mle_rating
```

## Testing

- `tests/style_policy/test_search.py` (pure): toy dict-tree — leaf at depth 0; a max node returns the
  best child + its move; a chance node returns the prob-weighted average (probs renormalized);
  terminal (empty `expand`) returns `leaf_value`; depth limit stops recursion.
- `policy_topk` (gated on `wdl_16M`): returns ≤k entries, all legal moves, probs descending and in
  (0,1]; a promotion position yields the queen move.
- `ExpectimaxBot` (gated): `choose_move` returns a legal move at depths 0,1,2; depth-0 equals
  `policy_topk`'s argmax; deterministic across two instances (same config) on a fixed position.
- Existing `play.py`/rating tests stay green.

## Compute note

Cost ≈ `width^depth` node-evaluations per move; depth-3 at width-4 ≈ ~84 forwards/move, so the
depth-3 rating run is slow (~tens of minutes at modest games). `--games-per-level` defaults low (25);
depths stream so shallow results arrive first. Sequential recursion (no batching) in v1.

## Out of scope

- Alpha-beta / transposition tables (don't prune cleanly over chance nodes); depth-varying width;
  adversarial-minimax opponent (we model the opponent as the human policy = expectimax); using
  base_64M's policy with the wdl value; batched leaf evaluation (a later optimization).

## Risks

- **Opening OOD**: the model is OOD on the first 4 plies (`skip_opening_plies=4`); leaving the opening
  book **on** keeps openings sane and isolates the search effect to the in-distribution middlegame.
- **Human-WDL leaf**: the bot optimizes expected *human-outcome* WDL, so it's positionally sound but
  tactically limited; if deeper search doesn't help, that (plus the value head's implicit lookahead)
  is the likely reason — a finding, not a bug.
- **Compute**: depth-3 is heavy; mitigated by low default games + per-depth streaming; batching is a
  future optimization.
- **Determinism**: top-K expansion is deterministic, so games vs a *sampling* Maia2 still vary (Maia
  samples); color-balanced in the rating run.
