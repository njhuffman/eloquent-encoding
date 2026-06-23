# Dev Log — eloquent-encoding / style_policy

Running log of substantive results and lessons. Newest entries on top. See
`docs/superpowers/plans/` for the implementation plan and `MEMORY`/CLAUDE notes for
the broader project context.

---

## 2026-06-22 — Data scaling concludes + capacity test (base validated; pivot to style)

**TL;DR:** Full scaling curve done. Data is the primary lever but now diminishing; the
6.8M base plateaus ~50.5% on 64M, ~2pts under Maia at equal model size. Base is validated
and Maia-competitive — stopping base scaling, pivoting to Phase-2 style. All numbers from
ONE consistent eval (fp32, n=10000 val seed42); bf16==fp32 confirmed (earlier JSON values
agreed).

| model | params | data | full-move top1 |
|---|---|---|---|
| base_4M | 6.8M | 4M | 30.5% |
| base_16M | 6.8M | 16M | 43.7% |
| base_32M | 6.8M | 32M | 48.5% |
| base_64M | 6.8M | 64M | 50.5% |
| base_32M_big | 22M | 32M | 50.4% |

- **Data gains/doubling: +6.6 → +4.8 → +2.0** (16M→32M→64M). Clearly diminishing.
- **Capacity (3.3x params @32M): +1.9pts.** 64M-small (50.5%) ≈ 32M-big (50.4%): data and
  capacity give similar returns here, both tapped out individually.
- **Maia-2 (rapid) on the SAME positions:** 52.5%. So at equal ~22M size, Maia leads ~3pts
  — purely its ~280x data. Last ~2pts need data+model TOGETHER (coupled), or a new approach.
- **Decision:** raw accuracy was never the goal; the base is competent + Maia-competitive,
  which is all the style work needs. Stop scaling the base; start Phase-2 (style).

**Build/infra notes:** j3_training_64M built (Jan+Feb, 800k games/stratum, sharded 10-way
+ shuffled merge; recipe dataset_generation/j3_training_64M.yaml). Container CUDA dropped
again post-run (NVML "Unknown Error") — needs `docker restart` before next GPU job; eval
runs on CPU. Re-eval harness: load checkpoint -> forward_policy -> argmax from/to vs human.

---

## 2026-06-21 — Maia-2 head-to-head on our val set (true apples-to-apples)

**TL;DR:** Ran Maia-2 (rapid) on the SAME 1500 val positions our model sees. Real gap is
only ~2pts — the closeness is genuine, not a test-set artifact.

| model | params | train data | full-move top1 | ≥5-legal |
|---|---|---|---|---|
| ours base_32M | 6.8M | 32M | 50.3% | 48.6% |
| Maia-2 rapid | ~22M | ~9B | 52.5% | 51.1% |

- Maia scores ~52.5% on OUR val ≈ its published ~53% → our val isn't easier; our 48-50%
  is legit. Single-legal positions only 1.1%; ≥5-legal cut preserves the ~2.5pt gap.
- Agreement: both 43.5%, neither 40.8%, ours-only 6.7%, maia-only 8.9% (partly different
  errors; we're not dominated).
- Caveats (both widen true gap slightly): we passed Maia elo_oppo=elo_self (we never
  stored opponent elo); our val is our exact training distribution (home-field).
- Method: isolated venv `pip install maia2` (+ undeclared deps: torch, chess, gdown,
  pyzstd, pyyaml, einops, sklearn, pandas). packed->FEN reconstruction validated (human
  move legal 300/300). Scripts: /tmp/maia_eval.py (venv), /tmp/ours_eval.py (main env).
- Follow-up: rerun vs the 22M bigger model (Maia's size) when it finishes — isolates
  data/recipe from capacity.

---

## 2026-06-21 — 32M run + scaling curve (4M → 16M → 32M)

**TL;DR:** Built `j3_training_32M` (sharded parallel build, ~2h) and trained `base_32M`
(identical recipe, warmup scaled to 1.6%). Held-out **full_top1 = 48.5%**. Scaling still
pays (16M→32M = +4.9pts) but with mild diminishing returns.

| data | val_loss | full_top1 | from_top1 | to_top1 |
|---|---|---|---|---|
| 4M | 2.31 | 30.5% | 45.4% | 69.2% |
| 16M | 1.85 | 43.6% | 55.8% | 78.3% |
| 32M | 1.66 | 48.5% | 59.9% | 80.7% |

Gains/doubling: ~6.5pts (4M→16M) → 4.9pts (16M→32M). Diminishing but not flat. 32M is
~4.5pts under Maia-2's ~53% with ~1/280th its data and a ~3-4x smaller model.

**Build notes:** sharded across 10 cores (1 elo stratum/shard) of 22; ~1h47m. Bottleneck
was python-chess parse (~60 games/s/shard, uniform). Merge shuffles rows in memory
(seed 12345). Dataset is a game-level superset of 16M; ply sampling differs from a
single-process build (shards use stratum_index=0) — irrelevant to the scaling result.
Recipe: `dataset_generation/j3_training_32M.yaml`. Run wandb `ax920dpz`. lr 2e-4/bf16
stable throughout.

**Data availability:** all 5 months counted (Jan–May 2025), each ~9.3–10.4M rapid games,
smallest bucket (1000–1099) ≥ ~490k/month. → 64M = Jan+Feb, ~200M possible from all 5.

**Next (open):** (1) 64M (Jan+Feb) to continue the data curve (~+3-4pts expected); (2)
bigger model at 32M to test whether capacity (6.8M params) is now the binding constraint
— leaning this, given diminishing data returns + small model. Do both to disambiguate.

---

## 2026-06-21 — Data-scaling probe (4M vs 16M)

**TL;DR:** 4M unique positions → **30.5%** full-move val; 16M → **43.6%**. A large ~13pt
gap → **not data-saturated at 16M**; scaling data helps. Equal-compute check shows the
gain tracks *unique data consumed* (not a compute/config artifact).

Same recipe as `base_16M` (bf16, lr 2e-4, warmup+cosine→0, batch 256, same val set),
only difference = 4M positions (1 epoch, warmup scaled to 250 = same 1.6%). Run wandb
`me5n9pr1`.

| run | data | steps | val_loss | full_top1 | from_top1 | to_top1 |
|---|---|---|---|---|---|---|
| base_4M | 4M | 15,625 | 2.31 | 30.5% | 45.4% | 69.2% |
| base_16M | 16M | 62,500 | 1.85 | 43.6% | 55.8% | 78.3% |

**Compute-confound check (equal steps):** at ~16k steps both have consumed ~4M unique
positions; they're ~tied — base_4M-final 30.5% (annealed) vs base_16M@16k 32.8% (not
annealed). So 16M's advantage isn't a config artifact; it comes from continuing to
consume fresh data (32.8%→43.6% over steps 16k→62.5k). Gain tracks unique-data-consumed,
no flattening by 16M. (Fully-airtight freshness-vs-steps test = 4M×4-epoch control; not
yet run.)

**Implication / next:** scaling data is well-motivated. Beyond 16M needs a data *build*
(file is 16M rows; raw PGNs on volume, pipeline exists). Suggested: an **8M subset run**
first (no build, ~1.4h) for a 3-point curve to confirm gains are still steep at 16M
before building 32M; if flattening, scale model instead.

---

## 2026-06-20 — First complete base policy run (`base_16M`)

**TL;DR:** Trained the first end-to-end `style_policy` base move-predictor. Held-out
**full-move top-1 = 43.6%** (elo-always-known). Recipe that works: **bf16 AMP, AdamW
lr 2e-4 with warmup+cosine**. Two failure modes found and fixed along the way (fp16
overflow; lr-too-high divergence).

### What the model is
- **Arch:** `BasePolicy` — 8-layer Transformer encoder over 64 square tokens + CLS,
  two pointer heads (from-square, then to-square conditioned on the chosen from), elo
  bucket-embedding conditioning. ~**6.8M params** (transformer is ~6.34M).
- **Objective:** `from_ce + to_ce`, masked cross-entropy over *legal* squares,
  label_smoothing 0.1 (smoothed over legal squares only). Supervised behavioral cloning
  of human moves. No JEPA/recon/aux losses. `to_head` is teacher-forced on the true
  from-square during training.
- **Data:** `j3_training_16M.h5` (16M positions, 1 epoch), val `j3_validation_1M.h5`.
  Reused existing on-disk jepa3-packed format (no rebuild). **Elo always known** (the
  deliberate "easier case" first; elo-dropout / optional-elo deferred).

### Training config (the one that worked)
- batch 256, AMP **bf16**, AdamW lr **2e-4**, **warmup 1000 + cosine decay to 0**,
  wd 0.01, grad-clip 1.0, ~62.5k steps. ~**2.8 h** on the laptop RTX 500 Ada (4GB; peak
  VRAM ~1.4GB, GPU ~92–98% util). Run: wandb `88zprkn9`.

### Result (held-out validation)
| metric | value |
|---|---|
| **full_top1 (full move)** | **43.6%** |
| from_top1 (which piece) | 55.8% |
| to_top1 (where, given piece) | 78.3% |
| val loss (from_ce+to_ce) | 1.85 (1.25 + 0.61) |

Checkpoints: `style_policy_checkpoints/base_16M/base_16M_stage_1.{best,}.pt`.

### Maia-2 comparison
Maia-2: ResNet+skill-attention, ~20–30M params (not officially reported), **9.15B
positions / 168.9M games**, ~**53%** full-move top-1. Ours reaches **43.6%** with a
**6.8M model on 16M positions** (~0.2% of Maia-2's data) — ~82% of its accuracy at a
tiny fraction of scale. Rough comparison (different val sets) but an encouraging baseline.

### Data-signal read (do we need more data?) — inconclusive
Val loss: 2.65(6k)→2.18(18k)→2.00(30k)→1.90(42k)→1.851(62.5k). Decelerating
(−0.47,−0.18,−0.10,−0.04,−0.007 per 12k) but still improving; no hard plateau. **The
tail flattening is confounded by cosine LR→0**, so saturation can't be read off this
single curve. No overfitting (val<train is a label-smoothing artifact: train loss is
smoothed, val isn't). Lean: entering diminishing returns; bottleneck (data vs the 6.8M
capacity) is undetermined. **Definitive test = 4M vs 16M, same schedule, compare final
val.**

### Failure modes found & fixed (the expensive lessons)
1. **fp16 AMP overflow → silent freeze.** First run's loss went NaN at ~step 10.8k and
   silently froze for thousands of steps (GradScaler skips NaN steps; GPU stayed 97%
   busy; flat metrics). Root cause = fp16 forward overflow (max 65504), NOT bad weights
   (max|w|=3.4) — proven: identical weights give loss NaN(fp16)/4.38(bf16)/2.63(fp32).
   **Fix:** default AMP → **bf16** (fp32 exponent range; no GradScaler). Added a
   fail-fast non-finite-loss guard so silent flatline can't recur. *(commit 5712863)*
2. **lr 5e-4 too high → divergence.** bf16 run then *diverged* (loss exploded
   3.1→260k, accuracy collapsed) at ~step 5.7k. bf16's lower mantissa precision makes
   aggressive LR more divergence-prone than fp16. warmup+cosine alone doesn't help (LR
   still ~5e-4 at the divergence point). **Fix:** peak lr **5e-4 → 2e-4**. *(commit
   73a9bbc)* The 2e-4 run sailed through the danger zone, loss monotonically down.

### Infra added this session
- W&B logging (optional `wandb:` block; auto GPU/CPU/VRAM). `full_top1` metric
  (honest joint from+to). Step-driven loop with periodic validation + resumable
  checkpoints (`--resume`) + warmup/cosine LR schedule. Vectorized board decode (~6×;
  GPU no longer starved).

### Open / next
- **Held-out-elo run** (the harder case; measure cost vs this 43.6% elo-known baseline).
- **Saturation probe** (4M vs 16M final val) to settle data-vs-capacity.
- **Phase 2: style buckets via EM** (the actual differentiator over Maia).

## 2026-06-23 — Opening book (per-elo-band, PGN-derived)

Built per-elo-band statistical opening books from `lichess_db_standard_rated_2025-01_tc_600_0.pgn.zst`
(`python -m scripts.build_opening_book --per-band-target 100000`), single-threaded/niced alongside
WDL training. 10 bands (1000–1900), 100k games each, first 24 plies, position-keyed by EPD
(transposition-merged), pruned to ≥0.1% support. Output: `/mnt/eloquence_bulk/databases/opening_book/band_*.json`
(~180K each, ~765–930 positions/band).

Sanity (start-position move %, elo-dependent as expected):
- 1200: e4 69% / d4 21% / Nf3 2%; after 1.e4 → e5 61%, d5 10%, c5 8%.
- 1800: e4 63% / d4 26% / c4 3%; after 1.e4 → e5 41%, c5 20% (Sicilian), e6 10% (French).

PolicyBot consults the book (off by default; `opening_book=` + `book_threshold=`, default 1%), samples
∝ human frequency while support ≥ threshold, then hands off to the model. Books live in the data dir
(not committed). Web-bot integration (shipping book JSON as a static asset) is future work.

## Queued / next work (backlog)

1. **WDL 16M gate eval** (immediate, when training finishes): A1 (WDL log-loss vs per-elo prior +
   policy full-top1 vs base_16M), phase-sliced. Then B1 (ΔV distribution human vs bot) + B2
   (disagreement-ΔV) per `docs/superpowers/specs/2026-06-23-value-head-evaluation-plan.md`.
2. **Web opening-book integration** (next web feature, AFTER the gate eval). Make the GitHub Pages
   bot use the opening book. Three pieces: (a) ship band JSONs as a static asset under
   `web/public/opening_book/` (~1.8MB total, all 10 bands) and fetch the elo→band file; (b) port
   `OpeningBook.lookup` to TS (EPD key, support threshold, sample ∝ counts; chess.js already
   present); (c) wire into `Engine`/`BoardPanel` to consult the book before the model, with the
   elo→band map + threshold control. KEY RISK: the EPD key must match byte-for-byte between
   python-chess `board.epd()` (build) and the chess.js `fen()`-minus-counters (play) — needs a
   parity fixture (like the boardToTensor/ONNX parity), or lookups silently miss. Full
   brainstorm→spec→plan when picked up (not yet designed).
3. **Deferred experiments:** opp-elo conditioning for the value head; phase-sliced WDL; expected-score
   target; checkpoint Elo curve + blunder-rate (incl. on base_4M→64M); move-match-by-rating-band.
