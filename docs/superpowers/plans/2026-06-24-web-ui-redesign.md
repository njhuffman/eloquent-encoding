# Web UI redesign (round C) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restyle the web bot to a modern-dark, responsive layout (settings toolbar on top; board + eval bar + info sidebar; reflows to one column on mobile) with no behavior changes.

**Architecture:** A token-driven global stylesheet (`styles.css`) holds the whole theme; components move from inline styles to semantic class names. A pure `boardSizeFor` clamp + a measured container give the board a responsive width; a `useMediaQuery` hook collapses the settings toolbar on mobile.

**Tech Stack:** TypeScript/React, react-chessboard 4.7.3, plain CSS (custom properties), vitest (node env).

## Global Constraints

- **Execution environment:** web `node_modules` is root-owned (container-installed); the HOST cannot run vitest/tsc/build. Run ALL web commands in the container:
  `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding/web && <cmd>'` (node 20). Run `git` on the HOST. Vitest is node-only, `include: ["src/**/*.test.ts"]`.
- Branch: `web-ui-redesign`.
- **No behavior changes** — every component keeps its exact props and logic; only markup/classes/styling change (plus the responsive board width). The existing **45 tests must stay green** as the regression net.
- The theme is **token-driven**: all colors/spacing/radius/font live in the `:root` block of `styles.css` (single source of truth). Theme = modern dark; layout = settings-on-top.
- Breakpoint: `@media (max-width: 700px)` for the mobile reflow.
- Git commit footer (verbatim, both lines):
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01VMxeVCfznS5H68W5SGyXFC`

---

## File Structure

- `web/src/styles.css` — CREATE: tokens + globals + base element styles + all component classes.
- `web/index.html` — MODIFY: viewport, lang, color-scheme.
- `web/src/main.tsx` — MODIFY: `import "./styles.css"`.
- `web/src/boardSize.ts` (+ `boardSize.test.ts`) — CREATE: pure `boardSizeFor` clamp.
- `web/src/useMediaQuery.ts` — CREATE: media-query hook (for the mobile settings collapse).
- `web/src/App.tsx` — MODIFY: layout shell (header + collapsible toolbar).
- `web/src/components/Controls.tsx` — MODIFY: `.toolbar` markup + segmented color toggle.
- `web/src/components/BoardPanel.tsx` — MODIFY: responsive board (ResizeObserver + `boardSizeFor`) + classed layout.
- `web/src/components/{WDLBar,ThinkingPanel,EloEstimate}.tsx` — MODIFY: classes/tokens (logic unchanged).

The class names defined in Task 1 are consumed by Tasks 2–4; each task still compiles independently (class names are plain strings; the build never depends on the CSS existing).

---

## Task 1: Foundation — stylesheet, viewport, board-size helper

**Files:**
- Create: `web/src/styles.css`, `web/src/boardSize.ts`, `web/src/boardSize.test.ts`
- Modify: `web/index.html`, `web/src/main.tsx`

**Interfaces:**
- Produces: `boardSizeFor(containerWidth: number, max?: number): number`; the full CSS class vocabulary (`.app`, `.app__header/__title/__subtitle/__status/__error`, `.toolbar-wrap`, `.toolbar`, `.control`/`.control__label`/`.control--dim`, `.check`, `.seg`/`.seg__opt`(`.is-active`), `.btn`/`.btn--primary`/`.btn--ghost`, `.board-area`, `.board-block`, `.board-stack`, `.board-host`, `.navrow`, `.status`, `.gameover`, `.sidebar`, `.card`/`.card__title`, `.wdlbar`/`.wdlbar__seg`(`.is-white/.is-black/.is-draw`), `.movelist`/`.move`/`.move__san`(`.is-chosen`)/`.move__bar`(`.is-chosen`)/`.move__pct`, `.empty-hint`, `.estimate__rating/__from/__chart/__bar`(`.is-map`)/`__labels/__caveat`).

- [ ] **Step 1: Write the failing test**

Create `web/src/boardSize.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { boardSizeFor } from "./boardSize";

describe("boardSizeFor", () => {
  it("caps at the max (default 480)", () => {
    expect(boardSizeFor(640)).toBe(480);
    expect(boardSizeFor(517.8)).toBe(480);
  });
  it("uses the container width (floored) when below the max", () => {
    expect(boardSizeFor(300)).toBe(300);
    expect(boardSizeFor(200.9)).toBe(200);
  });
  it("never returns less than 1 (guards 0 / negative / NaN-ish)", () => {
    expect(boardSizeFor(0)).toBe(1);
    expect(boardSizeFor(-50)).toBe(1);
  });
  it("honors a custom max", () => {
    expect(boardSizeFor(900, 600)).toBe(600);
  });
});
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding/web && npx vitest run src/boardSize.test.ts'`
Expected: FAIL — cannot resolve `./boardSize`.

- [ ] **Step 3: Create `web/src/boardSize.ts`**

```ts
// Responsive board edge length: the container width (floored), capped at `max`, never below 1
// (so a 0/NaN measurement can't reach react-chessboard).
export function boardSizeFor(containerWidth: number, max = 480): number {
  return Math.max(1, Math.min(Math.floor(containerWidth), max));
}
```

- [ ] **Step 4: Run the test**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding/web && npx vitest run src/boardSize.test.ts'`
Expected: PASS (4 cases).

- [ ] **Step 5: Create `web/src/styles.css`**

```css
:root {
  --bg: #1d1f25;
  --panel: #2a2d35;
  --panel-2: #22242b;
  --border: #383b44;
  --accent: #5b8def;
  --accent-strong: #4a7ad6;
  --good: #36a165;
  --text: #e6e8ec;
  --muted: #9aa0ab;
  --sq-light: #dde3ec;
  --sq-dark: #7a8aa0;
  --space-1: 4px; --space-2: 8px; --space-3: 12px; --space-4: 16px;
  --radius: 8px; --radius-sm: 5px;
  --font: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}

* { box-sizing: border-box; }
html { color-scheme: dark; }
body {
  margin: 0; background: var(--bg); color: var(--text);
  font-family: var(--font); font-size: 15px; line-height: 1.45;
}
h1, h2, h3 { margin: 0; font-weight: 650; }
a { color: var(--accent); }

/* layout shell */
.app { max-width: 880px; margin: 0 auto; padding: var(--space-4); }
.app__header { margin-bottom: var(--space-3); }
.app__title { font-size: 22px; letter-spacing: .2px; }
.app__subtitle { color: var(--muted); font-size: 13px; margin-top: 2px; }
.app__status { color: var(--muted); margin: var(--space-2) 0; }
.app__error { color: #ff6b6b; margin: var(--space-2) 0; }

/* settings toolbar (collapsible on mobile) */
.toolbar-wrap { background: var(--panel-2); border: 1px solid var(--border); border-radius: var(--radius); margin-bottom: var(--space-4); }
.toolbar-wrap > summary { list-style: none; cursor: pointer; padding: var(--space-2) var(--space-3); color: var(--muted); font-weight: 600; font-size: 13px; }
.toolbar-wrap > summary::-webkit-details-marker { display: none; }
.toolbar { display: flex; flex-wrap: wrap; gap: var(--space-4); align-items: flex-end; padding: var(--space-3); }
.control { display: flex; flex-direction: column; gap: 3px; font-size: 12px; }
.control__label { color: var(--muted); white-space: nowrap; }
.control--dim { opacity: .5; }

/* inputs */
input[type="range"] { width: 130px; accent-color: var(--accent); cursor: pointer; }
input[type="range"]:disabled { cursor: default; opacity: .6; }
input[type="checkbox"] { accent-color: var(--accent); width: 15px; height: 15px; }
.check { display: flex; align-items: center; gap: 6px; font-size: 13px; cursor: pointer; }

/* segmented toggle */
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: var(--radius-sm); overflow: hidden; }
.seg__opt { background: transparent; color: var(--text); border: 0; padding: 5px 12px; cursor: pointer; font: inherit; font-size: 13px; }
.seg__opt + .seg__opt { border-left: 1px solid var(--border); }
.seg__opt.is-active { background: var(--accent); color: #fff; }

/* buttons */
.btn { background: var(--panel); color: var(--text); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 6px 12px; font: inherit; font-size: 13px; cursor: pointer; }
.btn:hover:not(:disabled) { border-color: var(--accent); }
.btn:disabled { opacity: .45; cursor: default; }
.btn--primary { background: var(--accent); border-color: var(--accent); color: #fff; }
.btn--primary:hover:not(:disabled) { background: var(--accent-strong); }
.btn--ghost { background: transparent; }

/* board area */
.board-area { display: flex; gap: var(--space-4); align-items: flex-start; }
.board-block { flex: 1; min-width: 0; }
.board-stack { display: flex; gap: var(--space-2); }
.board-host { flex: 1; min-width: 0; max-width: 480px; }
.navrow { display: flex; align-items: center; gap: var(--space-2); flex-wrap: wrap; margin-top: var(--space-2); min-height: 28px; }
.status { color: var(--muted); font-size: 13px; }
.gameover { color: var(--muted); margin-top: var(--space-2); }
.sidebar { width: 260px; flex-shrink: 0; display: flex; flex-direction: column; gap: var(--space-3); }

/* card */
.card { background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius); padding: var(--space-3); }
.card__title { font-size: 14px; margin-bottom: var(--space-2); }

/* WDL bar */
.wdlbar { display: flex; flex-direction: column; width: 18px; border: 1px solid var(--border); border-radius: var(--radius-sm); overflow: hidden; flex-shrink: 0; }
.wdlbar__seg { display: flex; align-items: center; justify-content: center; font-size: 9px; }
.wdlbar__seg.is-white { background: #e9edf2; color: #222; }
.wdlbar__seg.is-black { background: #14161a; color: #cfd3da; }
.wdlbar__seg.is-draw { background: #5a5f6b; color: #fff; }

/* thinking panel / move list */
.movelist { display: flex; flex-direction: column; gap: 3px; }
.move { display: flex; align-items: center; gap: var(--space-2); font-size: 13px; }
.move__san { width: 52px; font-family: ui-monospace, "SF Mono", Menlo, monospace; }
.move__san.is-chosen { font-weight: 700; color: var(--good); }
.move__bar { height: 10px; border-radius: 2px; background: var(--accent); min-width: 1px; }
.move__bar.is-chosen { background: var(--good); }
.move__pct { color: var(--muted); font-size: 11px; margin-left: auto; }
.empty-hint { color: var(--muted); font-size: 13px; margin: 0; }

/* elo estimate */
.estimate__rating { font-size: 24px; font-weight: 700; color: var(--accent); }
.estimate__from { color: var(--muted); font-size: 12px; margin-bottom: var(--space-2); }
.estimate__chart { display: flex; align-items: flex-end; gap: 3px; height: 72px; }
.estimate__bar { flex: 1; min-height: 1px; background: var(--accent); border-radius: 2px 2px 0 0; }
.estimate__bar.is-map { background: var(--good); }
.estimate__labels { display: flex; gap: 3px; font-size: 9px; color: var(--muted); margin-top: 2px; }
.estimate__labels span { flex: 1; text-align: center; }
.estimate__caveat { color: var(--muted); font-size: 11px; margin-top: var(--space-1); }

/* responsive */
@media (min-width: 701px) {
  .toolbar-wrap > summary { display: none; } /* desktop: toolbar always shown (details open) */
}
@media (max-width: 700px) {
  .board-area { flex-direction: column; }
  .sidebar { width: 100%; }
}
```

- [ ] **Step 6: Import the stylesheet in `web/src/main.tsx`**

```tsx
import React from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./styles.css";
createRoot(document.getElementById("root")!).render(<App />);
```

- [ ] **Step 7: Update `web/index.html`** (add viewport / lang / color-scheme)

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="color-scheme" content="dark" />
    <title>Eloquent Bot</title>
  </head>
  <body><div id="root"></div><script type="module" src="/src/main.tsx"></script></body>
</html>
```

- [ ] **Step 8: Typecheck + build + full suite**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding/web && npx tsc --noEmit && npm run build && npx vitest run'`
Expected: tsc clean; build succeeds; all tests pass (46 now — the 45 existing + `boardSizeFor`).

- [ ] **Step 9: Commit**

```bash
git add web/src/styles.css web/src/boardSize.ts web/src/boardSize.test.ts web/index.html web/src/main.tsx
git commit -m "feat(web): dark theme tokens + global stylesheet, viewport, boardSizeFor"
```

---

## Task 2: App shell + settings collapse + Controls toolbar

**Files:**
- Create: `web/src/useMediaQuery.ts`
- Modify: `web/src/App.tsx`, `web/src/components/Controls.tsx`

**Interfaces:**
- Consumes: the CSS classes from Task 1.
- Produces: `useMediaQuery(query: string): boolean`; `App` renders `.app` shell with a collapsible `.toolbar-wrap`; `Controls` renders the `.toolbar` (same props as today).

- [ ] **Step 1: Create `web/src/useMediaQuery.ts`**

```ts
import { useEffect, useState } from "react";

// Returns whether `query` currently matches, updating on viewport changes.
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(
    () => typeof window !== "undefined" && window.matchMedia(query).matches,
  );
  useEffect(() => {
    const m = window.matchMedia(query);
    const on = () => setMatches(m.matches);
    on();
    m.addEventListener("change", on);
    return () => m.removeEventListener("change", on);
  }, [query]);
  return matches;
}
```

- [ ] **Step 2: Rewrite `web/src/App.tsx`**

```tsx
import React, { useEffect, useState } from "react";
import { useEngine } from "./useEngine";
import { BoardPanel } from "./components/BoardPanel";
import { Controls } from "./components/Controls";
import { useMediaQuery } from "./useMediaQuery";

export function App() {
  const { engine, error, books } = useEngine();
  const [botElo, setBotElo] = useState(1500);
  const [analysisElo, setAnalysisElo] = useState(1500);
  const [showAnalysis, setShowAnalysis] = useState(true);
  const [temperature, setTemperature] = useState(0.1);
  const [playerColor, setPlayerColor] = useState<"w" | "b">("w");
  const [gameStarted, setGameStarted] = useState(false); // locks the bot-elo control once a move is played

  // Settings toolbar collapses on mobile; open by default on desktop.
  const isMobile = useMediaQuery("(max-width: 700px)");
  const [settingsOpen, setSettingsOpen] = useState(!isMobile);
  useEffect(() => setSettingsOpen(!isMobile), [isMobile]);

  return (
    <div className="app">
      <header className="app__header">
        <h1 className="app__title">Eloquent Bot</h1>
        <div className="app__subtitle">A human-like chess bot — see how each rating would move, and estimate your own.</div>
      </header>
      {error && <p className="app__error">Failed to load model: {error}</p>}
      {!engine && !error && <p className="app__status">Loading model…</p>}
      <details className="toolbar-wrap" open={settingsOpen}
               onToggle={(e) => setSettingsOpen((e.currentTarget as HTMLDetailsElement).open)}>
        <summary>Settings ▾</summary>
        <Controls
          botElo={botElo} setBotElo={setBotElo} botEloLocked={gameStarted}
          analysisElo={analysisElo} setAnalysisElo={setAnalysisElo}
          showAnalysis={showAnalysis} setShowAnalysis={setShowAnalysis}
          temperature={temperature} setTemperature={setTemperature}
          playerColor={playerColor} setPlayerColor={setPlayerColor}
        />
      </details>
      <BoardPanel
        engine={engine} botElo={botElo} analysisElo={analysisElo} showAnalysis={showAnalysis}
        temperature={temperature} books={books} playerColor={playerColor}
        onGameStartedChange={setGameStarted}
      />
    </div>
  );
}
```

- [ ] **Step 3: Rewrite `web/src/components/Controls.tsx`** (same props, `.toolbar` markup + segmented toggle)

```tsx
import React from "react";

export function Controls({
  botElo, setBotElo, botEloLocked, analysisElo, setAnalysisElo,
  showAnalysis, setShowAnalysis, temperature, setTemperature, playerColor, setPlayerColor,
}: {
  botElo: number; setBotElo: (n: number) => void; botEloLocked: boolean;
  analysisElo: number; setAnalysisElo: (n: number) => void;
  showAnalysis: boolean; setShowAnalysis: (b: boolean) => void;
  temperature: number; setTemperature: (n: number) => void;
  playerColor: "w" | "b"; setPlayerColor: (c: "w" | "b") => void;
}) {
  return (
    <div className="toolbar">
      <label className="control">
        <span className="control__label">Bot elo: {botElo}{botEloLocked ? " 🔒" : ""}</span>
        <input type="range" min={600} max={2400} step={100} value={botElo} disabled={botEloLocked}
               onChange={(e) => setBotElo(Number(e.target.value))} />
      </label>
      <label className={"control" + (showAnalysis ? "" : " control--dim")}>
        <span className="control__label">Analysis elo: {analysisElo}</span>
        <input type="range" min={600} max={2400} step={100} value={analysisElo} disabled={!showAnalysis}
               onChange={(e) => setAnalysisElo(Number(e.target.value))} />
      </label>
      <label className="check">
        <input type="checkbox" checked={showAnalysis} onChange={(e) => setShowAnalysis(e.target.checked)} />
        Show analysis
      </label>
      <label className="control">
        <span className="control__label">Temperature: {temperature.toFixed(1)}</span>
        <input type="range" min={0.1} max={2.0} step={0.1} value={temperature}
               onChange={(e) => setTemperature(Number(e.target.value))} />
      </label>
      <div className="control">
        <span className="control__label">Play as</span>
        <div className="seg">
          <button className={"seg__opt" + (playerColor === "w" ? " is-active" : "")}
                  onClick={() => setPlayerColor("w")}>White</button>
          <button className={"seg__opt" + (playerColor === "b" ? " is-active" : "")}
                  onClick={() => setPlayerColor("b")}>Black</button>
        </div>
      </div>
    </div>
  );
}
```

(The color buttons are no longer `disabled` on the active side — the `.is-active` styling shows the selection; clicking the already-active color is a harmless no-op set.)

- [ ] **Step 4: Typecheck + build + full suite**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding/web && npx tsc --noEmit && npm run build && npx vitest run'`
Expected: tsc clean; build succeeds; all tests pass.

- [ ] **Step 5: Commit**

```bash
git add web/src/useMediaQuery.ts web/src/App.tsx web/src/components/Controls.tsx
git commit -m "feat(web): app shell + collapsible settings toolbar (modern dark)"
```

---

## Task 3: BoardPanel — responsive board + classed layout

**Files:**
- Modify: `web/src/components/BoardPanel.tsx`

**Interfaces:**
- Consumes: `boardSizeFor` (Task 1); the CSS classes from Task 1.
- Produces: the `.board-area` layout; a dynamically-sized board + matching eval bar. All existing props/logic/effects unchanged.

- [ ] **Step 1: Add the `boardSizeFor` import**

At the top of `web/src/components/BoardPanel.tsx`, with the other imports:

```tsx
import { boardSizeFor } from "../boardSize";
```

- [ ] **Step 2: Add the measured board width (ref + ResizeObserver)**

Immediately after the existing `const board = boardAtPly(history, viewPly);` / `const tip` / `const atTip` lines (around line 43–45), add:

```tsx
  const hostRef = useRef<HTMLDivElement>(null);
  const [hostWidth, setHostWidth] = useState(480); // fallback until measured
  useEffect(() => {
    const el = hostRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width;
      if (w) setHostWidth(w);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const boardWidth = boardSizeFor(hostWidth);
```

- [ ] **Step 3: Replace the returned JSX** (the whole `return (...)` block, currently lines ~222–249) with the classed, responsive layout:

```tsx
  return (
    <div className="board-area">
      <div className="board-block">
        <div className="board-stack">
          <WDLBar wdl={wdl} sideToMove={wdlStm} playerColor={playerColor} height={boardWidth} />
          <div className="board-host" ref={hostRef}>
            <Chessboard
              position={board.fen()}
              onPieceDrop={onDrop}
              onSquareClick={onSquareClick}
              arePiecesDraggable={!thinking}
              customSquareStyles={customSquareStyles}
              boardWidth={boardWidth}
              boardOrientation={boardOrientationOf(playerColor)}
            />
          </div>
        </div>
        <div className="navrow">
          <button className="btn btn--primary" onClick={newGame} disabled={thinking}>New game</button>
          <button className="btn btn--ghost" onClick={goBack} disabled={thinking || viewPly === 0}>◀</button>
          <button className="btn btn--ghost" onClick={goForward} disabled={thinking || atTip}>▶</button>
          <button className="btn btn--ghost" onClick={copyMoves} disabled={tip === 0}>{copied ? "Copied!" : "Copy moves"}</button>
          <span className="status">{status}</span>
        </div>
        {board.isGameOver() && <p className="gameover">Game over: {board.isCheckmate() ? "checkmate" : "draw"}</p>}
      </div>
      <div className="sidebar">
        {showAnalysis && <ThinkingPanel title="What would play here" moves={analysis} emptyHint="—" />}
        <EloEstimate estimate={estimate} bands={ELO_BANDS} moves={estimateMoves} minMoves={MIN_ESTIMATE_MOVES} />
      </div>
    </div>
  );
```

(Everything above the `return` — state, effects, `customSquareStyles`, `status`, the highlight rgba colors — is unchanged.)

- [ ] **Step 4: Typecheck + build + full suite**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding/web && npx tsc --noEmit && npm run build && npx vitest run'`
Expected: tsc clean; build succeeds; all tests pass (board logic unchanged).

- [ ] **Step 5: Commit**

```bash
git add web/src/components/BoardPanel.tsx
git commit -m "feat(web): responsive board (ResizeObserver + boardSizeFor) + classed layout"
```

---

## Task 4: Restyle leaf components (WDLBar, ThinkingPanel, EloEstimate)

**Files:**
- Modify: `web/src/components/WDLBar.tsx`, `web/src/components/ThinkingPanel.tsx`, `web/src/components/EloEstimate.tsx`

**Interfaces:**
- Consumes: the CSS classes from Task 1. All props/exports unchanged (incl. the exported `arrangeWDL`/`WDL` from WDLBar — its logic is untouched).

- [ ] **Step 1: Rewrite `web/src/components/WDLBar.tsx`** (keep `arrangeWDL`/`WDL` exactly; restyle only the render)

```tsx
import React from "react";

export type WDL = { loss: number; draw: number; win: number };
type Seg = { kind: "white" | "black" | "draw"; prob: number };

// Convert a side-to-move WDL into three segments ordered top->bottom, with the
// player's own color at the BOTTOM so the bar matches the flipped board.
export function arrangeWDL(
  wdl: WDL, sideToMove: "w" | "b", playerColor: "w" | "b",
): { top: Seg; mid: Seg; bottom: Seg } {
  const pWhite = sideToMove === "w" ? wdl.win : wdl.loss;
  const pBlack = sideToMove === "w" ? wdl.loss : wdl.win;
  const playerWhite = playerColor === "w";
  const bottom: Seg = playerWhite ? { kind: "white", prob: pWhite } : { kind: "black", prob: pBlack };
  const top: Seg = playerWhite ? { kind: "black", prob: pBlack } : { kind: "white", prob: pWhite };
  return { top, mid: { kind: "draw", prob: wdl.draw }, bottom };
}

export function WDLBar(
  { wdl, sideToMove, playerColor, height = 480 }:
  { wdl: WDL | null; sideToMove: "w" | "b"; playerColor: "w" | "b"; height?: number },
) {
  const a = wdl
    ? arrangeWDL(wdl, sideToMove, playerColor)
    : { top: { kind: "black", prob: 0 }, mid: { kind: "draw", prob: 1 }, bottom: { kind: "white", prob: 0 } } as
        { top: Seg; mid: Seg; bottom: Seg };
  const order: Seg[] = [a.top, a.mid, a.bottom];
  return (
    <div className="wdlbar" style={{ height }} title="White / draw / black win probability">
      {order.map((s, i) => (
        <div key={i} className={`wdlbar__seg is-${s.kind}`}
             style={{ flexGrow: Math.max(s.prob, 0.0001), flexBasis: 0 }}>
          {wdl && s.prob >= 0.08 ? Math.round(s.prob * 100) : ""}
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 2: Rewrite `web/src/components/ThinkingPanel.tsx`** (render as a `.card`)

```tsx
import React from "react";

type Move = { uci?: string; san: string; prob: number };

export function ThinkingPanel({ title, moves, highlightUci, emptyHint }: {
  title: string;
  moves: Move[];
  highlightUci?: string; // mark the move that was actually played (the bot's choice)
  emptyHint?: string;
}) {
  const max = moves.length ? moves[0].prob : 1;
  return (
    <div className="card">
      <h3 className="card__title">{title}</h3>
      {moves.length === 0 && <p className="empty-hint">{emptyHint ?? "—"}</p>}
      <div className="movelist">
        {moves.map((m) => {
          const chosen = !!m.uci && m.uci === highlightUci;
          return (
            <div key={m.uci ?? m.san} className="move">
              <span className={"move__san" + (chosen ? " is-chosen" : "")}>{chosen ? "▶ " : ""}{m.san}</span>
              <div className={"move__bar" + (chosen ? " is-chosen" : "")} style={{ width: `${(m.prob / max) * 100}%` }} />
              <span className="move__pct">{(m.prob * 100).toFixed(1)}%</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Rewrite `web/src/components/EloEstimate.tsx`** (render as a `.card`)

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
      <div className="card">
        <h3 className="card__title">Your estimated rating</h3>
        <p className="empty-hint">Play {need} more move{need === 1 ? "" : "s"} to estimate your rating.</p>
      </div>
    );
  }
  const max = Math.max(...estimate.posterior, 1e-9);
  return (
    <div className="card">
      <h3 className="card__title">Your estimated rating</h3>
      <div className="estimate__rating">≈ {Math.round(estimate.meanElo / 50) * 50}</div>
      <div className="estimate__from">from {moves} of your moves</div>
      <div className="estimate__chart">
        {bands.map((b, i) => (
          <div key={b} className={"estimate__bar" + (b === estimate.mapElo ? " is-map" : "")}
               style={{ height: `${(estimate.posterior[i] / max) * 70}px` }} />
        ))}
      </div>
      <div className="estimate__labels">{bands.map((b) => <span key={b}>{b}</span>)}</div>
      <p className="estimate__caveat">Indicative match, not a calibrated rating.</p>
    </div>
  );
}
```

- [ ] **Step 4: Typecheck + build + full suite**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding/web && npx tsc --noEmit && npm run build && npx vitest run'`
Expected: tsc clean; build succeeds; all tests pass (incl. the unchanged `WDLBar` `arrangeWDL` test).

- [ ] **Step 5: Commit**

```bash
git add web/src/components/WDLBar.tsx web/src/components/ThinkingPanel.tsx web/src/components/EloEstimate.tsx
git commit -m "feat(web): restyle WDL bar, analysis panel, rating estimate to dark theme"
```

---

## Self-review notes

- **Spec coverage:** tokens + global stylesheet + base element styles (T1); viewport/lang/color-scheme (T1); `boardSizeFor` + test (T1); app shell + settings-on-top + mobile collapse via `useMediaQuery` (T2); Controls toolbar + segmented toggle (T2); responsive board (ResizeObserver + clamp, eval bar matches) + classed layout (T3); leaf-component restyle (T4). 45 logic tests + `boardSizeFor` as the regression net; build/tsc gates each task.
- **Naming consistency:** `boardSizeFor`, `useMediaQuery`, and the CSS class vocabulary (`.app`, `.toolbar(-wrap)`, `.control(--dim)`, `.seg/.seg__opt.is-active`, `.btn(--primary/--ghost)`, `.board-area/.board-block/.board-stack/.board-host/.navrow`, `.sidebar`, `.card/.card__title`, `.wdlbar/.wdlbar__seg.is-*`, `.move*`, `.estimate__*`) are defined in T1 and used verbatim in T2–T4.
- **No behavior change:** every component keeps its props/logic; only markup/classes change, plus the board width going from fixed 480 to `boardSizeFor(measured)`.
- **Known follow-ups (out of scope):** ΔV coaching + WDL sparkline; light-theme toggle; animations.
```
