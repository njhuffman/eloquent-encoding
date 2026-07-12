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

## Pass 1 — RESULTS (2026-07-11): free lunch ✅

Target changed static `eval` → **depth-1 SF search cp** (tanh(cp/400), STM). The verbose static
`eval` command recomputes the full per-piece NNUE contribution table (~40 evals) per call (10.8ms);
depth-1 is 6× faster (1.6ms, 32M in ~1.7h @14 workers) and a better target — it resolves the
immediate tactic instead of scoring as if quiet. Labeled full 32M, 0 NA. Model = `multiband_ourdistill_human_nnue` (256/8, no-history, nnue_weight=1.0), trained 125k steps
(~5h), vs the `multiband_ourdistill_human` baseline. All metrics on held-out 2025-05 val, positions
freshly depth-1-labeled (no train contamination).

**1. Move prediction — NOT hurt (consistent small help).** Joint move-match, n=30k/band:
| band | baseline | nnue | Δ |
|---|---|---|---|
| 1000 | 46.7% | 47.0% | +0.3 |
| 1500 | 48.8% | 49.0% | +0.2 |
| 2000 | 49.5% | 50.2% | +0.7 |
nnue ≥ baseline at every band (CE also lower everywhere). Direction is consistent → real, not noise.

**2. WDL — NOT hurt (tiny help).** n=40k held-out: CE 0.7146→**0.7094**, acc 64.5%→**64.8%**.

**3. New eval capability — gained.** Held-out depth-1-eval decodability (Pearson r):
- baseline frozen-features probe: **0.878** (encoder already linearly carries eval)
- nnue frozen-features probe: **0.909** (+0.031, ~15 SE — co-training pushed eval further into features)
- nnue trained head, zero-shot: **0.935** (MSE 0.034)
So one forward pass now yields an objective eval at r≈0.94 — the human-vs-best divergence signal, no
Stockfish at inference.

**4. Concept features — unchanged.** material 0.904→0.907, mobility 0.809→0.811, hanging 0.412→0.426,
king_safety 0.583→0.565, isolated 0.299→0.317 — small bidirectional noise, no systematic shift
(unlike Maia-3 distillation which lifted 7/8). Eval content added *without* disturbing existing concepts.

**Verdict:** adding the SF-eval objective is a free lunch — no cost to human-play or WDL (both a hair
better), a genuine objective-eval head (r=0.935), encoder otherwise unchanged. Gate to Pass 2 PASSED.
Tools: `scripts/compare_nnue_multitask.py` (WDL+eval), `scripts/move_match.py`, `scripts/concept_probe.py`.

## Status
- Pass 1 complete, free-lunch result → Pass 2 gated open (128M-style wider run + SF-best-move head).
- Decisions locked: depth-1 SF-eval target, label full, eval-head only for Pass 1 (SF-move head deferred to Pass 2).
