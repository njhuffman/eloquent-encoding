# Web player-elo estimate (round B) — design

**Status:** approved design; precedes the implementation plan. Round B of the web improvements
(A. analysis & exploration controls — shipped; **B. player-elo estimate — this doc**; C. integrated
UI redesign). ΔV move-quality coaching and the WDL sparkline were considered for this round and
**deferred** (out of scope below).

## Goal

Estimate the player's strength from their moves: for each move the player made, the elo-conditioned
policy gives a probability under each elo band; accumulate across the player's moves into a
**posterior over bands**, shown as a small bar chart plus a headline rating. A "model as a mirror"
feature — which band's policy best matches how you play.

## The mechanism

The two-stage pointer policy is elo-conditioned, so for a position it scores every legal move per
elo band: `P(move | board, band) = P(from | board, band) · P(to | from, board, band)` (each
legality-masked softmax). For the player's moves `m₁…mₖ` (at the positions before them):

```
logL[band] = Σᵢ log P(mᵢ | board_before_i, band)
posterior[band] ∝ exp(logL[band])          # uniform prior over bands, then normalize
headline = Σ_band posterior[band] · band   # posterior-weighted mean elo
```

- **Only the player's own-color moves in the current line count.** Moves the player made *for the
  bot* during force-mode (round A) are the opposite color and are excluded.
- **Uniform prior**, so the posterior is the normalized likelihood across bands.
- **Caveat (documented in UI + spec):** the model is population behavior-cloning, so this is an
  indicative match, not a calibrated rating — it answers "which band's policy best predicts your
  move choices," which can skew vs. a true rating.

## Bands & display defaults

- Bands: **600–2400, step 200** → `[600,800,1000,1200,1400,1600,1800,2000,2200,2400]` (10 bars).
- Headline: posterior-weighted **mean elo**, rounded to the nearest 50.
- Minimum **3** player moves before showing the estimate (else a "play a few moves" hint).
- Shown always (independent of the round-A "Show analysis" toggle) — it's about the player, not the
  position.

## Components

### 1. Engine primitive — `engine.moveProbsByElo(board, move, elos)` (`web/src/inference/engine.ts`)
- Signature: `moveProbsByElo(board: Chess, move: { from: string; to: string }, elos: number[]): Promise<number[]>`.
- **Encode the board once** (the encoder is elo-independent), then for each elo: run `from_head` →
  `maskedSoftmax(fromLogits, legalFromMask(board), 1.0)` → take `pFrom[fromIdx]`; run `to_head` for
  `fromIdx` → `maskedSoftmax(toLogits, legalToMask(board, fromIdx), 1.0)` → take `pTo[toIdx]`; result
  `pFrom · pTo`. Returns one probability per elo (same order as `elos`).
- Reuses the existing `encode`, `this.run` (serialized queue), `maskedSoftmax`, `legalFromMask`/
  `legalToMask`, `squareToIndex`. Mirrors `chooseMove`'s structure. (Promotion is not separately
  modeled — the policy is from/to only — so a move's identity is its from/to, which is correct.)

### 2. Pure estimator — `web/src/eloEstimate.ts`
- `posteriorFromLogProbs(logProbsPerMove: number[][], elos: number[]): { posterior: number[]; meanElo: number; mapElo: number }`.
- Sums the per-move log-prob rows per band, softmaxes (max-subtracted) to a posterior, returns the
  posterior, the posterior-weighted mean, and the argmax (MAP) band. Empty input → uniform posterior
  (mean = average of `elos`). Pure; unit-tested.

### 3. BoardPanel integration (`web/src/components/BoardPanel.tsx`)
- Constant `ELO_BANDS = [600,800,1000,1200,1400,1600,1800,2000,2200,2400]`; `MIN_ESTIMATE_MOVES = 3`.
- A `useRef<Map<string, number[]>>` **cache**: key = the move-prefix string `history.slice(0, i+1).join(" ")` for player ply `i`, value = that move's per-band log-prob row. Different lines →
  different prefixes, so truncating rewinds (round A) are handled correctly; normal play adds one
  cache miss per move.
- State: `estimate: { posterior; meanElo; mapElo } | null` and `estimateMoves: number`.
- An effect keyed on **`[engine, history, playerColor]`** (NOT `viewPly` — navigation must not
  recompute) recomputes async + cancellable:
  - Get all moves with metadata: `const full = boardAtPly(history, history.length).history({ verbose: true })`.
  - For each `i` where `full[i].color === playerColor`: `key = history.slice(0, i+1).join(" ")`; if
    cached use the row, else `before = boardAtPly(history, i)`, `probs = await engine.moveProbsByElo(before, { from: full[i].from, to: full[i].to }, ELO_BANDS)`, `row = probs.map(p => Math.log(Math.max(p, 1e-9)))` (floor avoids `log(0)`), cache it.
  - `rows` = the player plies' rows. `setEstimateMoves(rows.length)`; if `rows.length >= MIN_ESTIMATE_MOVES` `setEstimate(posteriorFromLogProbs(rows, ELO_BANDS))` else `setEstimate(null)`.
- Render `<EloEstimate estimate={estimate} bands={ELO_BANDS} moves={estimateMoves} minMoves={MIN_ESTIMATE_MOVES} />` in the right column, under the analysis panel.

### 4. `web/src/components/EloEstimate.tsx`
- Props: `{ estimate: { posterior: number[]; meanElo: number; mapElo: number } | null; bands: number[]; moves: number; minMoves: number }`.
- `estimate === null`: a hint, e.g. `Play ${Math.max(minMoves - moves, 1)} more move(s) to estimate your rating`.
- Otherwise: headline `Estimated rating ≈ ${Math.round(meanElo / 50) * 50}` + `(from ${moves} moves)`,
  and a row of vertical bars — one per band, height ∝ `posterior[b] / max(posterior)`, the MAP band
  highlighted — with band labels beneath and a one-line "indicative, not a calibrated rating" note.
- Presentational; no inference.

## Data flow

```
line changes (a move is played/truncated):
  full = boardAtPly(history, len).history({verbose:true})
  rows = for each player-color ply i:
           key = history[0..i].join(" ")
           cache[key] ?? (engine.moveProbsByElo(boardAtPly(history,i), full[i].{from,to}, ELO_BANDS)
                           → log-floor → cache[key])
  estimate = rows.length >= 3 ? posteriorFromLogProbs(rows, ELO_BANDS) : null
  <EloEstimate estimate moves={rows.length} .../>

navigation (◀/▶): viewPly changes only → effect (keyed on history) does NOT refire → no recompute
```

## Testing

- **`eloEstimate.test.ts`** (pure): a move strongly favored by one band pulls the posterior + MAP to
  it; uniform log-prob rows → uniform posterior and mean = average of bands; mean is the
  posterior-weighted sum; empty input → uniform.
- **Engine `moveProbsByElo`** (in `engine.node.test.ts`, deterministic, no new fixture): for the
  start position at one elo, `moveProbsByElo(board, {from,to}, [elo])[0]` equals the product of the
  masked-softmax `from`/`to` probabilities computed via the existing `distributions(board, elo)`
  path (cross-checks the new method against established code), and all returned values are in `(0,1]`.
- Existing suites stay green; component wiring verified by `tsc --noEmit` + `npm run build` (no DOM
  test env — logic lives in the pure estimator + engine primitive).

## Out of scope (this round)

- ΔV move-quality coaching and the WDL sparkline (deferred; both reuse `engine.value` and can be a
  later round).
- Calibrating the estimate against true ratings; per-move "you played like ~1700 here" annotations.
- The integrated visual redesign / mobile (round C) — though `EloEstimate` should be a self-contained
  component the redesign can restyle/relocate.

## Risks

- **Inference volume:** 10 bands × player moves. Mitigated by encode-once-per-position (the encoder
  is the costly part and is elo-independent) + the per-prefix cache (one cache miss per normal move).
  Recompute is async/cancellable and keyed off `history`, so nav and slider changes don't trigger it.
- **`log(0)`:** a legal played move always has positive masked prob, but floor to `1e-9` before
  `log` to be safe.
- **Interpretability:** documented caveat — population-BC means the number is indicative, not a
  calibrated Elo.
