# Does a human-move-predictor chess policy drift, and can GAIL fix it?

**Date:** 2026-07-10 · **Branch:** `pgx-gail-scoping`

## Headline

**A policy that samples our human-move predictor does not meaningfully drift from human play — from a discriminator's point of view — at its operating temperature. So there is nothing meaningful for GAIL (or any imitation reward) to improve.**

At the temperature you'd actually use to hit a target strength (T≈0.7 → ~1440 Elo), the bot's
self-play positions are near-indistinguishable from real human positions to a *strong* discriminator
across every input representation we tried (~0.06 AUC over the noise floor), and its move choices are
already the maximum-likelihood human moves. The apparent "drift" only shows up under loose sampling
(T=1.0), and lowering temperature — which you do anyway for strength — removes it for free.

## Background & motivation

Behavior-cloning agents can drift off the expert's state distribution in rollout (compounding error).
Our move predictor is a BC policy, so the worry was: sampling it in self-play wanders into positions
no human reaches, degrading play and human-likeness. GAIL (a discriminator that rewards staying on the
human state-manifold) is the classic fix. We built the full stack to test it.

## What we built (all committed, reusable)

- **pgx GPU self-play** (~132k env-steps/s on a 4GB card), with validated `our↔pgx` **board and action
  bridges** and **GPU legal-move masking** (`pgx_bridge.py`, `pgx_action.py`, `pgx_rollout.py`).
- **GAIL stage (b): reward-weighted regression** (`gail_rwr.py`) — frozen encoder + policy head,
  discriminator reward, KL-to-predictor leash. (Debugged to stability; notable trap: `where(mask, 0·-inf)`
  gives NaN *gradients* — use finite mask values in any differentiated masked term.)
- **Drift / move-match diagnostics** (`selfplay_drift_seeded.py`, `drift_ceiling.py`, `move_match.py`).

## The finding, step by step

**1. Drift is real only at loose sampling.** Seeded from human ply-10 positions, the predictor at
**T=1.0** drifts monotonically: discriminator AUC 0.513 → 0.785 over +2 → +24 bot plies (floor 0.548).

**2. RWR "reduces" it — but that was a strength/temperature confound.** RWR cut +24 drift 0.785 → 0.690
*at fixed T=1.0*. But RWR also made the bot **stronger at every temperature** (e.g. 1244 → 1323 Elo at
T=1.0; up to **1806 Elo** when trained at T=0.8), and stronger play stays on-manifold. Lowering the
*plain predictor's* temperature does the same thing for free.

**3. At matched strength, the plain predictor is as human-like or more.** Comparing at ~1440 Elo —
predictor@T=0.7 (1436) vs RWR@T=0.8 (1450):

| horizon | predictor@0.7 | RWR@0.8 |
|---|---|---|
| +8  | **0.578** | 0.637 |
| +16 | **0.618** | 0.652 |
| +24 | 0.629 | **0.605** |

RWR only edges it at the longest horizon; the predictor is more human-like elsewhere.

**4. No hidden headroom — a stronger discriminator finds *less*, not more.** Gap over floor for
predictor@0.7 vs human at +24, across discriminator strength and input representation:

| discriminator | human-vs-bot AUC | floor | gap |
|---|---|---|---|
| weak (mean-pooled encoder) | 0.609 | 0.498 | 0.112 |
| strong (mean-pooled encoder) | 0.602 | 0.530 | 0.072 |
| strong (raw board planes) | 0.556 | 0.498 | 0.058 |
| attention over all 65 tokens | 0.572 | 0.508 | **0.064** |

The gap is ~0.06 and **representation-invariant**: mean-pooling wasn't hiding signal, raw pixels and a
spatial attention-pool over the full encoding agree. The bot's positions are genuinely near-human.

**5. The move axis is closed too.** Move-match vs human on 20k held-out band-1500 positions:

| model | joint move-match % | CE |
|---|---|---|
| predictor (256/8) | 50.9 | 1.580 |
| RWR (256/8) | 50.1 | 1.585 |
| 128M-big (384/12) | 52.4 | 1.519 |

RWR ties the predictor (marginally worse) — expected, since **cross-entropy on human moves *is* the
move-distribution-matching objective**, so the predictor is optimal on human states and GAIL can't beat
it. The only headroom is **encoder capacity** (128M-big +1.5%), not GAIL.

## Conclusion

On a human-CE predictor, **GAIL is redundant for human-likeness on both the state-occupancy and
move-distribution axes** (five independent confirmations). Two practical takeaways:

- **Human-like bot recipe:** temperature-controlled sampling of the predictor. Temperature is the
  dominant, free lever for *both* strength and human-likeness.
- **The only lever for *more* human-likeness is a better/bigger encoder** — which loops back to the
  project's core: the encoder-as-world-model is both the deliverable and the lever.

GAIL's one real product was a **stronger** bot (up to 1806 Elo) — irrelevant for "human-like at a target
level," but a hint that self-play RL on this encoder *can* push strength (see the RL-strength direction).
The GAIL infrastructure is retained and would do real work on a **non-human base policy**.
