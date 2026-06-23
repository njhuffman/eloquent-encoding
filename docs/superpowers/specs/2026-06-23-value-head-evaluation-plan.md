# Value-head & bot evaluation plan

**Status:** living agenda (notes from design discussion). Most items run *after* the WDL value
head finishes training; a few (strength rating, checkpoint curves) apply to any checkpoint.

**Purpose:** decide — empirically, before writing any value-weighted training loss — whether the
human-realized WDL value head is a *real guidance signal* or just an echo of the policy, whether
the bot plays *like its target elo*, and whether training improves it *beyond CE loss*. Organized
by the question each eval answers.

## Background facts these evals lean on

- **WDL is a martingale under human play.** V(s) = P(W/D/L | s, elo) is, by construction, the
  expected outcome over the moves humans play from s. So the value of the *average* (and roughly
  the *modal/high-probability*) human move ≈ V(s) → expected per-move ΔV ≈ 0 if calibrated. The
  *typical* move should barely move WDL; large ΔV on a high-probability move flags either a sharp
  human-pitfall position or a value-function miscalibration.
- **"Slow, no jumps" is only half true.** Per-move ΔV is peaked near 0 (bulk) but has a *fat tail*
  of jumps concentrated at blunders / tactics / conversions; a single game's V travels from ~0.5
  to the final {0,1} via those steps + jumps. The object to measure is the *distribution* of
  per-move ΔV (peak + tails), per elo.
- **CE is blind to rare blunders** (a catastrophic move is ~1 position in millions of CE), but it
  loses games — so strength / blunder metrics expose progress CE hides.
- **Population BC is often stronger than the average individual** (regresses to the modal,
  usually-correct move; averages out idiosyncratic blunders — cf. Maia argmax > nominal rating).
- **ΔV perspective:** value flips sides after a move (opponent to move). Always compute ΔV from a
  consistent (mover's) perspective when comparing across a ply.

---

## A. Is the value head itself any good?

**A1 — WDL vs per-elo prior + calibration (the go/no-go gate).** *Already implemented:*
`scripts/eval_wdl.py` (`prior_logloss_from_results` is per-elo-bucket). WDL log-loss must beat the
per-elo-bucket marginal prior (head learned position-dependent value, not base rate), and the
policy full-move top-1 must not regress vs `base_16M`. Add a calibration curve (predicted P(win)
in bins vs realized rate). **Outcome:** pass → the value head is real; proceed to B. Fail → tune
`value_loss_weight`, reconsider CLS vs mean-pool input, or whether joint training is right.

## B. Does the value head carry guidance signal, or just echo the policy?

**B1 — ΔV distribution: human games vs bot games (per elo).**
- *Method:* over many positions, compute per-move ΔV (mover perspective). Plot the distribution
  for (a) real human games and (b) bot self-play, per elo.
- *Outcomes:* mean ≈ 0 on human games = calibrated (martingale); systematic drift = miscalibrated.
  Bot ΔV ≈ human ΔV → value trajectory already human-like → **value head redundant with the
  policy**. Bot has a *fatter negative tail* → bot blunders more than its target humans → value
  head **catches real errors** → guidance is useful. Bot *smoother* than humans → over-stable
  (e.g. low temperature) → less human.

**B2 — Disagreement ΔV (the sharpest check).**
- *Method:* restrict to positions where the model's top move ≠ the human's move. Record
  `ΔV_human`, `ΔV_model`, and bucket by `ΔV_human`. (One eval pass: get model top move + human
  move, apply each, value both resulting boards.) Reuses `forward_value`.
- *Outcomes (sign of ΔV_model − ΔV_human on the disagreement set):*
  - ≈ 0 → divergences are equally-valued alternatives (benign; value head not revealing much).
  - model > human → model *outplays* the target humans where they differ (usually = avoids human
    mistakes; for a faithful 1800 this is a fidelity cost).
  - model < human → the model's deviations are *errors* — **the king-walk cluster; the set
    value-guidance/search should shrink.**
- *Cross-tab against `ΔV_human`* — "are they different precisely when a human would blunder?":
  at human-blunder positions (ΔV_human very negative), does the model disagree *and* pick higher
  ΔV (corrects the mistake)? Uncorrelated → divergences are noise. Agrees at blunder positions →
  model memorized the human mistake.
- *Prediction:* a `model > human` cluster concentrated at human-blunder positions (population-BC
  self-correction) **and** a smaller `model < human` cluster = the model's own characteristic
  blunders. The size of that second cluster tells you whether value-guidance is worth building:
  tiny → mostly redundant; substantial → real lever.

## C. Does the bot play like its target elo?

**C1 — Move-match accuracy by rating band (the Maia diagonal).** For each target-elo setting,
measure move-match (full_top1) against held-out human games from each rating band. **"Plays like
1800" ⇔ accuracy peaks when the band matches the target.** Cheap; reuses the eval harness with
per-band validation slices. Absolute number ceilings ~50–55% (humans aren't deterministic); the
signal is the *peak on the diagonal*, not the raw %.

## D. Is training improving the bot beyond CE loss?

**D1 — Relative strength across checkpoints (self-play Elo curve).** Round-robin matches between
checkpoints (step 10k/20k/40k/… or the data-scaling set base_4M→64M) via the `play_match` harness;
fit Elo with **Ordo/BayesElo** (error bars). Elo climbing while CE/top-1 plateau = the subtle
"fewer blunders / better tail" learning CE hides. Use **SPRT** for "is variant A > B" decisions.
*Ready experiment:* run the existing base_4M/16M/32M/64M checkpoints through this to give the
data-scaling study a strength lens (it was only compared on CE).

**D2 — Blunder rate vs training steps.** Per move, a referee flags catastrophic moves; track the
rate across checkpoints. Referee options: **objective** (Stockfish eval-drop > threshold / walks
into mate — fine as an *evaluation* referee even though we avoid it for training value);
**human-derived** (the model's own WDL craters — the ΔV signal, circular but informative as a
trend); **rule-based** (early king-walk, hangs queen — objective, targets observed failure modes).

## E. How strong is the bot, in absolute terms?

Strength is always games → Elo, cheapest → most authoritative:
- **Internal round-robin + Ordo** (relative, no anchor) — have the harness.
- **vs Stockfish at limited strength** (Skill Level / UCI_LimitStrength) — rough absolute anchor
  (±~100; Stockfish's own calibration is approximate).
- **Lichess BOT account** — the gold standard; rated like a human from real games at the training
  time control (600+0). The `Player.choose_move` seam was designed for a `lichess-bot` adapter.
- Caveats: Elo is pool- and time-control-specific (Lichess 1800 ≠ FIDE/chess.com); need many
  games for tight CIs (~1000+ for ±30).

---

## Sequencing

1. WDL training finishes → **A1 gate** (already built).
2. If gate passes → **B1 (ΔV distribution, human vs bot)** then **B2 (disagreement ΔV)** — these
   decide whether the value head is a real guidance signal *before* any value-weighted loss is
   written. This is the key fork: redundant → don't build the dual-loss; useful → build it (the
   one-sided "shave the catastrophic ΔV tail" formulation, per the design discussion).
3. **C1 (move-match diagonal)** — verify the elo conditioning is faithful (cheap, anytime).
4. **D1/D2** — retroactively on the data-scaling checkpoints, and going forward as a training
   progress signal.
5. **E** — strength rating when a bot is worth publishing (Lichess) or comparing rigorously.

## Notes on the value-weighted-loss decision (downstream of B)

The dual-loss idea (imitation CE + a value term) is **advantage-weighted behavior cloning**
(π_human·exp(β·A), tilt-toward-good-moves moved into training). Open design points settled in
discussion: penalty should be **one-sided** (suppress the catastrophic negative-ΔV tail) for a
*strength* goal, or **distribution-matching** (reproduce the human ΔV tail, including jumps) for a
*realism* goal — NOT symmetric |ΔV| minimization, which over-smooths past real human play. With a
human-WDL value it reduces *gross* blunders but is too blunt for mate-in-one-while-winning (that
needs objective value, which trades away elo-fidelity). β is a tunable dial, not a slide to
superhuman; ceiling is the value function's quality. Build only if B shows a non-trivial
`model < human` disagreement cluster.
