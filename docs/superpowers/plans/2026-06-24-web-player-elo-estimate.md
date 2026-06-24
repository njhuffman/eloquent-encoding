# Web player-elo estimate (round B) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Estimate the player's strength by scoring each of their moves under every elo band with the elo-conditioned policy, accumulating into a posterior over bands shown as a bar chart + headline rating.

**Architecture:** A new engine primitive scores a specific move per elo (encode once, run the two heads per band, legality-masked). A pure estimator turns per-move log-probs into a posterior + mean/MAP elo. `BoardPanel` caches per-ply rows (keyed by move-prefix), recomputes on line change over the player's own-color moves, and renders an `EloEstimate` chart.

**Tech Stack:** TypeScript/React, chess.js v1, onnxruntime, vitest (node env).

## Global Constraints

- **Execution environment:** web `node_modules` is root-owned (container-installed); the HOST cannot run vitest/tsc/build. Run ALL web commands in the container:
  `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding/web && <cmd>'` (node 20). Run `git` on the HOST. Vitest is node-only, `include: ["src/**/*.test.ts"]` (no DOM/testing-library — logic goes in pure helpers).
- Branch: `web-elo-estimate`.
- `P(move | board, band) = P(from|board,band)·P(to|from,board,band)`, each a legality-masked softmax (temperature 1.0). All ORT runs go through the engine's serialized queue (`this.run`).
- Only the **player's own-color** moves in the current line count. Estimate recomputes on **line change**, NOT on navigation (`viewPly`).
- Bands `[600,800,1000,1200,1400,1600,1800,2000,2200,2400]`; headline = posterior-weighted mean rounded to nearest 50; minimum **3** player moves to display.
- Uniform prior over bands; floor probabilities to `1e-9` before `log`.
- Git commit footer (verbatim, both lines):
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01VMxeVCfznS5H68W5SGyXFC`

---

## File Structure

- `web/src/inference/engine.ts` — MODIFY: add `moveProbsByElo` (+ import `squareToIndex`).
- `web/src/inference/engine.node.test.ts` — MODIFY: add a deterministic cross-check test.
- `web/src/eloEstimate.ts` (+ `eloEstimate.test.ts`) — CREATE: pure `posteriorFromLogProbs`.
- `web/src/components/EloEstimate.tsx` — CREATE: presentational bar chart + headline.
- `web/src/components/BoardPanel.tsx` — MODIFY: per-ply cache, estimate effect, render `EloEstimate`.

---

## Task 1: Engine primitive `moveProbsByElo`

**Files:**
- Modify: `web/src/inference/engine.ts`
- Test: `web/src/inference/engine.node.test.ts`

**Interfaces:**
- Produces: `engine.moveProbsByElo(board: Chess, move: { from: string; to: string }, elos: number[]): Promise<number[]>` — `P(move)` per elo (same order), each `pFrom·pTo` with legality-masked softmax at temperature 1.0. Encodes the board once.

- [ ] **Step 1: Write the failing test**

In `web/src/inference/engine.node.test.ts`, add these imports at the top (alongside the existing ones) if not present:

```ts
import { squareToIndex } from "./boardTensor";
import { maskedSoftmax } from "./sample";
import { legalFromMask, legalToMask } from "./legal";
```

Then add this test (a new `describe`, reusing the same int8 graph URLs as the existing parity tests):

```ts
describe("Engine.moveProbsByElo", () => {
  it("equals the masked from/to product from distributions, and is a probability", async () => {
    const eng = await Engine.load(ort as any, {
      encode: "public/encode_int8.onnx",
      fromHead: "public/from_head_int8.onnx",
      toHead: "public/to_head_int8.onnx",
      valueHead: "public/value_head_int8.onnx",
    }, { nEloBuckets: fixtures.n_elo_buckets });
    const board = new Chess(); // start position
    const elo = 1500;
    const { fromLogits, toLogits } = await eng.distributions(board, elo);
    const fromIdx = squareToIndex("e2"), toIdx = squareToIndex("e4");
    const pFrom = maskedSoftmax(fromLogits, legalFromMask(board), 1.0)[fromIdx];
    const tl = await toLogits(fromIdx);
    const pTo = maskedSoftmax(tl, legalToMask(board, fromIdx), 1.0)[toIdx];
    const expected = pFrom * pTo;
    const [got] = await eng.moveProbsByElo(board, { from: "e2", to: "e4" }, [elo]);
    expect(got).toBeCloseTo(expected, 6);
    expect(got).toBeGreaterThan(0);
    expect(got).toBeLessThanOrEqual(1);
  });
});
```

(`Chess`, `ort`, and `fixtures` are already imported in this test file.)

- [ ] **Step 2: Run it to confirm it fails**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding/web && npx vitest run src/inference/engine.node.test.ts'`
Expected: FAIL — `eng.moveProbsByElo is not a function`.

- [ ] **Step 3: Implement `moveProbsByElo`**

In `web/src/inference/engine.ts`, change the `boardTensor` import to also bring in `squareToIndex`:

```ts
import { boardToTensor, indexToSquare, squareToIndex } from "./boardTensor";
```

Add this method to the `Engine` class (e.g. after `chooseMove`):

```ts
  // Probability the policy assigns to a SPECIFIC move (from→to), at each elo. Encodes once
  // (the encoder is elo-independent), then runs the two heads per elo with legality-masked
  // softmax. Used by the player-elo estimate.
  async moveProbsByElo(board: Chess, move: { from: string; to: string }, elos: number[]): Promise<number[]> {
    const { squares: sq } = await this.encode(board);
    const fromIdx = squareToIndex(move.from), toIdx = squareToIndex(move.to);
    const fromMask = legalFromMask(board), toMask = legalToMask(board, fromIdx);
    const out: number[] = [];
    for (const elo of elos) {
      const eloT = this.elo(elo);
      const fl = (await this.run(this.fh, { squares: sq, elo_idx: eloT }))["from_logits"].data;
      const pFrom = maskedSoftmax(fl, fromMask, 1.0)[fromIdx];
      const fsqT = new this.ort.Tensor("int64", BigInt64Array.from([BigInt(fromIdx)]), [1]);
      const tl = (await this.run(this.th, { squares: sq, from_sq: fsqT, elo_idx: eloT }))["to_logits"].data;
      const pTo = maskedSoftmax(tl, toMask, 1.0)[toIdx];
      out.push(pFrom * pTo);
    }
    return out;
  }
```

- [ ] **Step 4: Run the test**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding/web && npx vitest run src/inference/engine.node.test.ts'`
Expected: PASS (the new test + the existing parity tests).

- [ ] **Step 5: Commit**

```bash
git add web/src/inference/engine.ts web/src/inference/engine.node.test.ts
git commit -m "feat(web): Engine.moveProbsByElo — P(move) per elo band"
```

---

## Task 2: Pure estimator `posteriorFromLogProbs`

**Files:**
- Create: `web/src/eloEstimate.ts`
- Test: `web/src/eloEstimate.test.ts`

**Interfaces:**
- Produces: `posteriorFromLogProbs(logProbsPerMove: number[][], elos: number[]): { posterior: number[]; meanElo: number; mapElo: number }`. Sums per-move log-prob rows per band, softmaxes to a posterior (uniform prior), returns the posterior, posterior-weighted mean, and argmax (MAP) band. Empty input → uniform.

- [ ] **Step 1: Write the failing test**

Create `web/src/eloEstimate.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { posteriorFromLogProbs } from "./eloEstimate";

const L = Math.log;
const bands = [1000, 1500, 2000];

describe("posteriorFromLogProbs", () => {
  it("equal log-probs → uniform posterior, mean = average band", () => {
    const r = posteriorFromLogProbs([[L(0.3), L(0.3), L(0.3)]], bands);
    expect(r.posterior[0]).toBeCloseTo(1 / 3, 6);
    expect(r.posterior[1]).toBeCloseTo(1 / 3, 6);
    expect(r.posterior[2]).toBeCloseTo(1 / 3, 6);
    expect(r.meanElo).toBeCloseTo(1500, 6);
  });

  it("a move one band loves pulls the posterior + MAP to it", () => {
    const r = posteriorFromLogProbs([[L(0.05), L(0.8), L(0.05)]], bands);
    expect(r.mapElo).toBe(1500);
    expect(r.posterior[1]).toBeGreaterThan(r.posterior[0]);
    expect(r.posterior[1]).toBeGreaterThan(r.posterior[2]);
  });

  it("accumulates across moves", () => {
    const r = posteriorFromLogProbs([[L(0.1), L(0.2), L(0.7)], [L(0.1), L(0.2), L(0.7)]], bands);
    expect(r.mapElo).toBe(2000);
    expect(r.meanElo).toBeGreaterThan(1500);
  });

  it("empty input → uniform posterior", () => {
    const r = posteriorFromLogProbs([], bands);
    for (const p of r.posterior) expect(p).toBeCloseTo(1 / 3, 6);
    expect(r.meanElo).toBeCloseTo(1500, 6);
  });
});
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding/web && npx vitest run src/eloEstimate.test.ts'`
Expected: FAIL — cannot resolve `./eloEstimate`.

- [ ] **Step 3: Create `web/src/eloEstimate.ts`**

```ts
// Posterior over elo bands from per-move log-probabilities (uniform prior over bands).
// logProbsPerMove[i][b] = log P(move i | band b). Returns the normalized posterior, the
// posterior-weighted mean elo, and the argmax (MAP) band.
export function posteriorFromLogProbs(
  logProbsPerMove: number[][], elos: number[],
): { posterior: number[]; meanElo: number; mapElo: number } {
  const n = elos.length;
  const logL = new Array(n).fill(0);
  for (const row of logProbsPerMove) {
    for (let b = 0; b < n; b++) logL[b] += row[b];
  }
  const m = Math.max(...logL);
  const w = logL.map((x) => Math.exp(x - m)); // softmax (max-subtracted for stability)
  const s = w.reduce((a, b) => a + b, 0);
  const posterior = w.map((x) => x / s);
  let mapElo = elos[0], best = -Infinity, meanElo = 0;
  for (let b = 0; b < n; b++) {
    meanElo += posterior[b] * elos[b];
    if (posterior[b] > best) { best = posterior[b]; mapElo = elos[b]; }
  }
  return { posterior, meanElo, mapElo };
}
```

- [ ] **Step 4: Run the test**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding/web && npx vitest run src/eloEstimate.test.ts'`
Expected: PASS (4 cases).

- [ ] **Step 5: Commit**

```bash
git add web/src/eloEstimate.ts web/src/eloEstimate.test.ts
git commit -m "feat(web): posteriorFromLogProbs — band posterior + mean/MAP elo"
```

---

## Task 3: `EloEstimate` component + BoardPanel integration

**Files:**
- Create: `web/src/components/EloEstimate.tsx`
- Modify: `web/src/components/BoardPanel.tsx`

**Interfaces:**
- Consumes: `posteriorFromLogProbs` (Task 2), `engine.moveProbsByElo` (Task 1), `boardAtPly` (existing `gameNav`).
- Produces: `EloEstimate` component; the estimate wired into `BoardPanel` (cache + effect + render).

- [ ] **Step 1: Create `web/src/components/EloEstimate.tsx`**

```tsx
import React from "react";

export function EloEstimate(
  { estimate, bands, moves, minMoves }:
  { estimate: { posterior: number[]; meanElo: number; mapElo: number } | null;
    bands: number[]; moves: number; minMoves: number },
) {
  if (!estimate) {
    const need = Math.max(minMoves - moves, 1);
    return (
      <div style={{ minWidth: 210 }}>
        <h3 style={{ marginBottom: 6 }}>Your estimated rating</h3>
        <p style={{ color: "#999", margin: 0, fontSize: 13 }}>
          Play {need} more move{need === 1 ? "" : "s"} to estimate your rating.
        </p>
      </div>
    );
  }
  const max = Math.max(...estimate.posterior, 1e-9);
  return (
    <div style={{ minWidth: 210 }}>
      <h3 style={{ marginBottom: 2 }}>Your estimated rating</h3>
      <div style={{ fontSize: 22, fontWeight: 700 }}>≈ {Math.round(estimate.meanElo / 50) * 50}</div>
      <div style={{ color: "#777", fontSize: 12, marginBottom: 6 }}>from {moves} of your moves</div>
      <div style={{ display: "flex", alignItems: "flex-end", gap: 3, height: 72 }}>
        {bands.map((b, i) => (
          <div key={b} style={{
            flex: 1, height: `${(estimate.posterior[i] / max) * 70}px`, minHeight: 1,
            background: b === estimate.mapElo ? "#2e7d32" : "#4a90d9", borderRadius: "2px 2px 0 0",
          }} />
        ))}
      </div>
      <div style={{ display: "flex", gap: 3, fontSize: 9, color: "#777", marginTop: 2 }}>
        {bands.map((b) => <span key={b} style={{ flex: 1, textAlign: "center" }}>{b}</span>)}
      </div>
      <p style={{ color: "#999", fontSize: 11, marginTop: 4 }}>Indicative match, not a calibrated rating.</p>
    </div>
  );
}
```

- [ ] **Step 2: Wire `BoardPanel.tsx` — imports + constants + state**

In `web/src/components/BoardPanel.tsx`, add to the imports:

```tsx
import { EloEstimate } from "./EloEstimate";
import { posteriorFromLogProbs } from "../eloEstimate";
```

Below the existing `const MOVE_DELAY_MS = 650;` (module scope), add:

```tsx
const ELO_BANDS = [600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400];
const MIN_ESTIMATE_MOVES = 3;
```

With the other `useState`/`useRef` hooks in the component, add:

```tsx
  const eloCacheRef = useRef<Map<string, number[]>>(new Map()); // move-prefix → per-band log P(move)
  const [estimate, setEstimate] = useState<{ posterior: number[]; meanElo: number; mapElo: number } | null>(null);
  const [estimateMoves, setEstimateMoves] = useState(0);
```

- [ ] **Step 3: Add the estimate effect**

Add this effect alongside the other effects in `BoardPanel` (it is keyed on `[engine, history, playerColor]` — NOT `viewPly`, so navigation never recomputes):

```tsx
  // Player-elo estimate: score each of the player's own-color moves in the current line under every
  // band, accumulate into a posterior. Per-ply rows cached by move-prefix so normal play computes
  // only the new ply and truncating rewinds reuse unchanged prefixes.
  useEffect(() => {
    if (!engine) { setEstimate(null); setEstimateMoves(0); return; }
    let cancelled = false;
    (async () => {
      const full = boardAtPly(history, history.length).history({ verbose: true });
      const rows: number[][] = [];
      for (let i = 0; i < full.length; i++) {
        if (full[i].color !== playerColor) continue;
        const key = history.slice(0, i + 1).join(" ");
        let row = eloCacheRef.current.get(key);
        if (!row) {
          const before = boardAtPly(history, i);
          const probs = await engine.moveProbsByElo(before, { from: full[i].from, to: full[i].to }, ELO_BANDS);
          if (cancelled) return;
          row = probs.map((p) => Math.log(Math.max(p, 1e-9)));
          eloCacheRef.current.set(key, row);
        }
        rows.push(row);
      }
      if (cancelled) return;
      setEstimateMoves(rows.length);
      setEstimate(rows.length >= MIN_ESTIMATE_MOVES ? posteriorFromLogProbs(rows, ELO_BANDS) : null);
    })().catch(() => {});
    return () => { cancelled = true; };
  }, [engine, history, playerColor]);
```

- [ ] **Step 4: Render `EloEstimate` in the right column (always visible)**

Find the right-column JSX (currently the analysis panel gated by `showAnalysis`):

```tsx
      {showAnalysis && (
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <ThinkingPanel title="What would play here" moves={analysis} emptyHint="—" />
        </div>
      )}
```

Replace it with a column that always renders the estimate, gating only the analysis panel:

```tsx
      <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
        {showAnalysis && <ThinkingPanel title="What would play here" moves={analysis} emptyHint="—" />}
        <EloEstimate estimate={estimate} bands={ELO_BANDS} moves={estimateMoves} minMoves={MIN_ESTIMATE_MOVES} />
      </div>
```

- [ ] **Step 5: Typecheck, build, full suite**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding/web && npx tsc --noEmit && npm run build && npx vitest run'`
Expected: tsc clean; build succeeds; all tests pass (engine + estimator tests from Tasks 1–2 included).

- [ ] **Step 6: Commit**

```bash
git add web/src/components/EloEstimate.tsx web/src/components/BoardPanel.tsx
git commit -m "feat(web): player-elo estimate (band posterior chart + headline rating)"
```

---

## Self-review notes

- **Spec coverage:** move-prob-per-band primitive (T1); posterior math (T2); player-color-only + prefix cache + recompute-on-line-change-not-nav + min-3-moves gating + chart/headline + always-visible (T3). Bands/headline/min-moves match the spec's defaults; `log` floor 1e-9 applied at the call site (T3).
- **Naming consistency:** `moveProbsByElo`, `posteriorFromLogProbs`, `ELO_BANDS`, `MIN_ESTIMATE_MOVES`, `eloCacheRef`, `estimate`/`estimateMoves`, `EloEstimate` used identically across tasks.
- **Known follow-ups (out of scope):** ΔV coaching + WDL sparkline (later round); calibration; the round-C visual redesign.
```
