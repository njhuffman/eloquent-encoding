# Joint per-band heads (co-adapting encoder) — design

**Status:** validated in brainstorming; precedes the implementation plan.

## Goal

Train one shared, **elo-input-free** encoder jointly with **10 per-band policy heads** (one per
100-Elo band, 1000–1900) and test whether the *co-adapting* encoder extends the strength ladder
past the frozen-encoder ceiling — especially pushing the high end above the ~1800 that frozen
heads saturated at. Scale: **7M arch (256/8), `wdl_training_16M`, ~3h**.

## Why

Frozen-encoder per-band heads gave **no move-match gain** but a **real low-temp strength ladder**
(bands 1000/1500/1900 → ~1567/1780/1828 Maia-rapid, monotonic, vs the elo-conditioning's flat
~1780) — see [[diagonal-findings]]. The high end saturated at `base_64M`'s ~1800 feature ceiling.
Hypothesis: letting the encoder **co-adapt to the per-band heads** (rather than reading frozen
features) raises the high-end ceiling → a fuller 1000→1900 strength range. The encoder still takes
**no elo input** (bands are selected by routing to a head), honoring [[elo-agnostic-encoder]].

## Comparison baseline (no extra run)

`wdl_16M` — the *same* 256/8 arch on the *same* `wdl_training_16M` with elo-conditioning + value
head. Clean same-arch/same-data A/B: joint per-band heads vs elo-embedding conditioning.

## Architecture — `MultiBandPolicy` (new)

- Shared `BoardEncoder` (256/8, **no elo input**) — identical to `BasePolicy`'s, so its weights are
  checkpoint-compatible.
- `heads`: `ModuleList` of **10 `BandHead`** (reused; `FromHead`/`ToHead` with `elo_dim=0`), one per
  band. Sample → head index `clamp((elo−1000)//100, 0, 9)`.
- `value_head`: **one shared elo-conditioned `WDLHead`** (as in `wdl_16M`) — kept for comparability
  and because value co-training may have helped the high-elo diagonal. Reads `cls`.

## Training — `train_multiband`

Adapts `train_one_stage`: same loader (`PackedMoveDataset`, mixed batches), bf16 AMP,
`torch.compile(encoder)`, cosine LR + warmup, periodic val + resumable checkpoint, final save.
Per batch: encode once → `(cls, squares)`; route by band — for each band `g`, the samples in that
band get `heads[g]` from/to masked-CE; total policy loss = size-weighted mean over bands; plus the
shared value CE (`value_head(cls, elo_idx)`). Every head updates every step on its share (mixed
batches), so no staleness; the encoder is pulled by all heads' gradients.

**Saves** (so existing tooling is reused unchanged):
1. Joint checkpoint `{architecture, bands, model}`.
2. An **encoder checkpoint** `BasePolicy`-loadable (`architecture` + state with `encoder.*` +
   `value_head.*` keys) — `BandHeadBot`/`eval_band_head_row` load it `strict=False` and use only
   `encode`.
3. **Per-band `BandHead` exports** `{band_head, d_model, hidden, source_checkpoint=<encoder ckpt>,
   band}` — directly consumable by `rate_band_heads` / `eval_band_head_row`.

## Eval / success

- **Strength ladder (primary):** `scripts/rate_band_heads.py` on the exported heads at temp 0.1 and
  1.0 → compare to the frozen ladder (1567/1780/1828) and to `wdl_16M`'s flat conditioning.
  **Win = a steeper, monotonic ladder with the high end pushing past ~1800.**
- **Move-match (secondary):** per-band-head move-match across bands (reuse `eval_band_head_row`'s
  `spec` column per exported head), vs `wdl_16M`'s diagonal.

## Out of scope

Smooth elo interpolation between heads; per-band value heads; deploying to the web bot; the 64M /
bigger-arch versions (follow-ups if the 16M test is promising).

## Risks

- **Per-band data (1.6M/band at 16M)** is smaller than the frozen heads' base_64M encoder saw (64M);
  if the joint encoder is too weak the high-end may still saturate — that itself is an informative
  result (→ try 64M).
- Routing loop over 10 bands per batch — cheap (heads are tiny), but verify correctness (size-weighted
  mean equals the un-routed mean when all samples share a band).
- compile + the per-band routing: keep the routing outside the compiled region (only the encoder is
  compiled), as in `train_one_stage`.
