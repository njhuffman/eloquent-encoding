# How strong a policy can we get from the frozen human world-model?

**Date:** 2026-07-10/11 · **Branch:** `pgx-gail-scoping`

## Headline

**Policy-only self-play RL does not climb past the base human predictor (~1900). The frozen
encoder is a solid ~1900-level 1-ply chess *guide* — competitive with Stockfish's NNUE at
1-ply — and getting past that is a *search* problem, not a model problem.**

## Setup

Warm-start a band head from the human move-predictor (band 2100 = strongest prior), self-play to
termination in pgx, REINFORCE on game outcome, frozen encoder (`scripts/selfplay_rl.py`). Strength
measured vs the Maia2 ladder at matched temperature.

## Results

**Policy-only self-play RL — no climb.**

| variant | T=0.5 | T=0.3 |
|---|---|---|
| baseline predictor | 1869 | 1951 |
| frozen-encoder RL (18 iters) | 1865 | 1943 |
| **unfrozen-encoder RL** | **1319** | **1255** |

Frozen RL plateaus *at* the base predictor. Unfreezing the encoder **collapses** it (86% draws in
training — naive REINFORCE treats a draw as "safe" and wrecks the pretrained features).

**Searchless bracket vs Maia2** (`rate_value_bot.py`, `rate_stockfish_1ply.py`):

| bot | search | Elo |
|---|---|---|
| our policy (human moves) | ~0-ply | ~1900 |
| our value head as evaluator | 1-ply | ~1869 |
| Stockfish NNUE | 1-ply | ~1985 |
| Stockfish NNUE | 8-ply | ~2559+ (100% vs Maia-1900) |

## Conclusion

- **Search depth is the strength lever**, not the model or the evaluator quality at shallow depth.
  Stockfish gains ~575 Elo from depth 1→8; at 1-ply *everything* clusters ~1900–2000.
- **Our human world-model is a competitive 1-ply guide** — its value head (~1869) is close to
  Stockfish's NNUE (~1985) at 1-ply. Using the world-model as an *evaluator* instead of a *policy*
  doesn't break the ceiling, because 1-ply is the limiter.
- **The ceiling is not the encoder per se** — it's (a) the human training target (imitation caps at
  human strength) and (b) the absence of search. The clean way to climb is **search on top of the
  world-model** (MCTS/AlphaZero), which is exactly what lifts Stockfish from 1985→2559.

*(Correction to an earlier claim: "1-ply NNUE ~2400–2800" was wrong — 1-ply can't see the reply.
DeepMind's ~2900 searchless net distilled *deep* Stockfish search, not a 1-ply eval.)*

## Open probe

To separate "encoder capacity" from "human-target ceiling": train a fresh head on the **frozen**
encoder to predict **Stockfish NNUE eval** (a strong target), then rate it as a 1-ply value bot. If
it beats the human-value bot (~1869), the encoder's features *can* support stronger play and the
human targets were the limiter; if not, the frozen features cap it.
