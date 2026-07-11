# Multi-task encoder: human moves + WDL + Stockfish NNUE eval

**Goal:** make the encoder learn *both* human play and objective (NNUE) evaluation, so one model
gives — in a single forward pass, no Stockfish at inference — the human move distribution, WDL, and
the objective eval (→ human-vs-best divergence). Fixes the frozen-encoder cap (r~0.89 predicting SF)
by making NNUE eval a *training objective* instead of a frozen probe.

Not trying to *beat* NNUE (Stockfish would have) — trying to *absorb* it.

## Pass 1 — controlled help/hurt test (short, ~half day)

Gates Pass 2. Does adding the NNUE-eval objective help or hurt the human tasks, and does the
encoder actually learn the eval when it's a training target?

- **Model:** 256/8, no-history — identical to `multiband_ourdistill_human` (the control).
- **Heads:** existing per-band human-move heads + WDL, **+ one NNUE-eval head** predicting the
  static NNUE eval (`sf_static_cp`), regressed as value = tanh(cp/400) (STM).
- **Loss:** `move_ce + wdl_ce + λ·nnue_eval_loss` (λ≈0.5–1).
- **Data:** the same 32M as the baseline, **fully labeled** with static NNUE (~1h, `static_nnue_label.py`).
- **Metrics (the point):**
  1. human move-match (from/to/joint) vs baseline — help/hurt on the policy
  2. WDL CE/acc vs baseline — does objective-value teaching improve human-value prediction
  3. NNUE-eval head accuracy (corr with SF) — does it blow past the frozen-probe r~0.89
  4. concept probes — did the shared features get richer or diluted
- **Decision:** move-match/WDL neutral-or-better AND NNUE-eval accuracy ≫0.89 → Pass 2. Move-match
  drops meaningfully → capacity dilution on 256/8 → motivates going wider in Pass 2.

## Pass 2 — big run if promising (128M-style, multiday)

- **Scale:** wider/deeper (384/12 or 512/8) + 128M samples (NNUE-label ~3.5–4.7h). Capacity headroom
  so the tasks don't compete.
- **+ SF-best-move / action-value head** → the payoff: one pass gives human play, best play, and the
  gap. (Add only if Pass 1 shows NNUE-eval meaningfully helps.)
- **Deliverable:** a self-contained hybrid analysis engine (human + objective + divergence), and the
  substrate for the commentary bridge.

## Implementation pieces

1. `static_nnue_label.py` — parallel static-NNUE labeler → row-aligned sidecar (`sf_static_cp`).
2. NNUE-eval head on `MultiBandPolicy` + multi-task loss in `multiband_train.py` (NNUE loss on
   labeled rows; the sidecar joins to the training h5 by row).
3. Config `multiband_ourdistill_human_nnue.yaml` (= human config + nnue head/target + sidecar path).

## Status
- Static-NNUE throughput benchmarked: 628/s single, ~7.5–10k/s @ 12–16 workers (32M ~1h).
- Decisions locked: static NNUE target, label full, eval-head only for Pass 1.
