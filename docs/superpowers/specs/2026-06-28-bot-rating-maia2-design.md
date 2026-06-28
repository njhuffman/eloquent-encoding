# Bot rating via Maia2 — design

**Status:** approved design; precedes the implementation plan.

## Goal

Estimate an approximate **Elo for one bot configuration** by playing many games against **Maia2 (rapid)**
at several rating levels, then fitting an anchored rating (≈ lichess-rapid scale). Maia2 is the right
anchor: like our bot it's a human-move model, so the matchup is apples-to-apples (vs Stockfish's
un-human limited-strength).

## Context (verified)

- Reuse `style_policy/play.py`: `Player.choose_move(board)->Move` seam, `PolicyBot` (our elo-conditioned
  model bot with temperature + opening book), `play_game` (handles draw claims + ply cap), `play_match`
  (fixed-color N-game match → W/D/L).
- Maia2 0.9 installed in the container; weights cached at `maia2_models/` (gitignored). API:
  `m = model.from_pretrained(type="rapid", device="gpu")`; `prep = inference.prepare()`;
  `move_probs, win_prob = inference.inference_each(m, prep, fen, elo_self, elo_oppo)` where `move_probs`
  is `{uci: prob}` over legal moves. **~2.5 ms/call on GPU**, so hundreds of games take minutes.
- Maia2 rapid valid elo range ~1100–2000; we use **1100/1300/1500/1700/1900** as anchors.

## Components

### 1. `style_policy/maia2_bot.py` — `Maia2Bot(Player)` + loader
- `load_maia2(type="rapid", device="gpu") -> (model, prep)` — loads once (offline-OK once weights are
  cached; falls back `device="cpu"` if GPU unavailable).
- `class Maia2Bot(Player)`: `__init__(self, model, prep, self_elo, opp_elo=None, seed=0)`
  (`opp_elo` defaults to `self_elo`); the model/prep are **shared** across per-level bots.
  - `choose_move(board)`: `mp, _ = inference.inference_each(model, prep, board.fen(), self_elo, opp_elo)`;
    keep only entries whose uci is a legal move, renormalize, and **sample** one with the bot's RNG
    (sampling — not argmax — so Maia plays at its *nominal* rating; argmax is stronger than the label and
    would break the anchor). Fallback to a uniform-random legal move if `mp` is empty after filtering.
  - `reset()`: no-op (Maia is stateless per position; the RNG intentionally advances across games for
    variety).

### 2. `style_policy/rating.py` — rating math (pure, unit-tested)
- `expected_score(anchor, rating) = 1/(1+10**((anchor-rating)/400))`.
- `implied_rating(score, anchor) = anchor + 400*log10(score/(1-score))`, with `score` clamped to
  `(eps, 1-eps)` so 0%/100% don't blow up.
- `score_ci(wins, draws, losses) -> (score, lo, hi)` — score = (wins+0.5·draws)/n with a normal-approx
  SE → 95% interval (clamped to [0,1]).
- `mle_rating(rows) -> (rating, se)` where `rows = [(anchor, n, score), …]`: single-parameter Newton on
  the log-likelihood `Σ n_i[s_i·ln E_i + (1-s_i)·ln(1-E_i)]`, `E_i = expected_score(anchor_i, rating)`.
  Gradient `Σ n_i(ln10/400)(s_i-E_i)`, Hessian `-(ln10/400)^2 Σ n_i E_i(1-E_i)`;
  `se = sqrt(1/((ln10/400)^2 Σ n_i E_i(1-E_i)))`. Init at the n-weighted mean of per-anchor implied
  ratings; clamp the result to `[min_anchor-1200, max_anchor+1200]` for degenerate all-win/all-loss data.

### 3. `scripts/rate_bot.py` — the runner
- Args: `--checkpoint` (default `style_policy_checkpoints/base_64M/base_64M_stage_1.pt`), `--elo`
  (default 1500), `--temperature` (default 0.1), `--book/--no-book` (default book on, using the per-elo
  opening book), `--levels` (default `1100 1300 1500 1700 1900`), `--games-per-level` (default 100),
  `--device` (default cuda), `--max-plies` (default 300), `--seed` (default 0), `--out` (optional JSON).
- Build the `PolicyBot` under test once; `load_maia2` once. For each level: build `Maia2Bot(level)`; play
  **color-balanced** — `play_match(white=bot, black=maia, n=games//2)` and `play_match(white=maia,
  black=bot, n=games//2)` — and tally the **bot's** W/D/L (bot-white: bot win = white win; bot-black: bot
  win = black win). Per-game variety comes from the advancing RNGs (both sides sample); seeds set at
  construction for reproducibility.
- Compute per-level score + `implied_rating` (+ CI), then the global `mle_rating` over all levels. Print
  a table (level, W/D/L, score, implied ±) and the headline **rating ± 95% CI**, plus a monotonicity
  note (score should fall as the Maia level rises). Write the results JSON if `--out` given.

## Data flow

```
PolicyBot(checkpoint, elo, temp, book)      maia, prep = load_maia2()
for R in levels:
  maia_R = Maia2Bot(maia, prep, self_elo=R)
  bot W/D/L  <- play_match(bot, maia_R, games/2) + play_match(maia_R, bot, games/2)   # color-balanced
  score_R = (W + 0.5 D)/games ;  implied_R = implied_rating(score_R, R)
rating, se = mle_rating([(R, games, score_R) for R in levels])
print table + (rating ± 1.96·se)
```

## Testing

- `tests/style_policy/test_rating.py` (pure, no Maia): `expected_score(R,R)==0.5` and monotonic;
  `implied_rating(0.5,A)==A`, >A for score>0.5; `mle_rating` **recovers a known rating** from synthetic
  rows built as `s_i = expected_score(anchor_i, 1600)` → ~1600; `se` shrinks as `n` grows;
  `score_ci` bounds within [0,1].
- `Maia2Bot` smoke (gated: skip if `maia2`/weights unavailable): from a few positions `choose_move`
  returns a **legal** move; sampling with a fixed seed is deterministic.
- Runner smoke: 1 level × 2 games on CPU/GPU completes and produces a finite rating (gated on Maia2).
- Existing `play.py` tests stay green.

## Out of scope

- The elo-**sweep** calibration curve (we chose single-config; the runner is parametrized so a loop over
  `--elo` adds it later); lichess-bot/online play; multi-bot Ordo/BayesElo round-robin; SPRT.

## Risks

- **Maia calibration vs sampling**: Maia must **sample** (not argmax) to sit at its nominal rating —
  enforced in `Maia2Bot`. Documented that the result is a Maia/lichess-rapid rating, ±CI by game count,
  not FIDE.
- **Determinism/variety**: both sides sample with advancing RNGs; `reset()` deliberately does not reseed,
  so the N games of a match differ. Color-balanced to cancel first-move advantage.
- **Offline runtime**: weights are cached in `maia2_models/`; `from_pretrained` should load locally
  without network. The container's DNS fix (public nameserver) is only needed to (re)download weights,
  not at run time — verify local load in implementation.
- **Degenerate scores** (a level the bot beats/loses 100%): `implied_rating` clamps and `mle_rating`
  bounds the estimate; more games or wider level spread tightens it.
