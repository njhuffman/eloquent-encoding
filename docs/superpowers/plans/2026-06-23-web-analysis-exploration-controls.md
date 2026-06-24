# Web analysis & exploration controls (cluster A) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the bot's single-elo, gameRef-only board with a navigable game (ply back/forward), force-move by playing at a rewound position, a fixed bot elo locked at game start, and a separate scrub-able "analysis elo" driving one live top-moves panel + board highlight.

**Architecture:** Pure helpers (`gameNav.ts`) own the game-as-SAN-list model. `BoardPanel` keeps the authoritative SAN list + a `viewPly` cursor; the displayed board is replayed from the list. A move at the viewed ply truncates the rest and continues; the bot auto-replies only when a move hands it the turn. Bot strength (`botElo`) is separate from the always-editable `analysisElo`.

**Tech Stack:** TypeScript/React, react-chessboard 4.7.3, chess.js v1, vitest (node env).

## Global Constraints

- **Execution environment:** web `node_modules` is root-owned (container-installed); the HOST cannot run vitest/tsc/build. Run ALL web commands in the container:
  `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding/web && <cmd>'` (node 20).
  Run `git` on the HOST. Vitest config is node-only, `include: ["src/**/*.test.ts"]` (so `.test.ts`, not `.test.tsx`; no DOM/testing-library — logic goes in pure helpers).
- Branch: `web-analysis-controls`.
- WDL order (loss=0, draw=1, win=2), side-to-move perspective; all ORT runs go through the engine's serialized queue (unchanged — we only call existing `topMoves`/`engine.value`/`bookOrModelMove`).
- **Defaults (settled):** `showAnalysis` default `true`; bot elo locked by **disabling** its control while a move has been played (re-enabled on New Game).
- The **WDL bar uses `botElo`** (the game's fixed strength); the **analysis panel + board highlight use `analysisElo`**; the **bot's moves use `botElo`**.
- Git commit footer (verbatim, both lines):
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01VMxeVCfznS5H68W5SGyXFC`

---

## File Structure

- `web/src/gameNav.ts` (+ `gameNav.test.ts`) — CREATE: pure game-model helpers (`boardAtPly`, `truncateAndPlay`, `shouldBotReply`).
- `web/src/components/BoardPanel.tsx` — REWRITE (Task 2): SAN-list + `viewPly` model, ◀/▶ + arrow-key nav, move-for-side-to-move (force), single analysis panel; then MODIFY (Task 3): split elo + showAnalysis + gameStarted.
- `web/src/undo.ts` + `web/src/undo.test.ts` — DELETE (Task 2): replaced by ply navigation.
- `web/src/App.tsx` — MODIFY (Task 3): `botElo`/`analysisElo`/`showAnalysis`/`gameStarted` state.
- `web/src/components/Controls.tsx` — MODIFY (Task 3): bot-elo (lockable) + analysis-elo sliders + show-analysis checkbox.
- `web/src/clickMove.ts` — unchanged; caller now passes `board.turn()` as the mover color (the existing Black-mover test already covers this).

---

## Task 1: Pure game-navigation helpers

**Files:**
- Create: `web/src/gameNav.ts`
- Test: `web/src/gameNav.test.ts`

**Interfaces:**
- Produces:
  - `boardAtPly(history: string[], ply: number): Chess` — a fresh `Chess` with `history[0:ply]` applied.
  - `truncateAndPlay(history: string[], ply: number, move: {from:string;to:string;promotion?:string}): string[] | null` — `history[0:ply]` + the move's SAN, or `null` if illegal.
  - `shouldBotReply(board: Chess, botColor: "w"|"b"): boolean` — `!board.isGameOver() && board.turn()===botColor`.

- [ ] **Step 1: Write the failing test**

Create `web/src/gameNav.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { Chess } from "chess.js";
import { boardAtPly, truncateAndPlay, shouldBotReply } from "./gameNav";

describe("boardAtPly", () => {
  const h = ["e4", "e5", "Nf3", "Nc6"];
  it("replays the first `ply` moves into a fresh board", () => {
    expect(boardAtPly(h, 0).fen()).toBe(new Chess().fen());
    expect(boardAtPly(h, 2).history()).toEqual(["e4", "e5"]);
    expect(boardAtPly(h, 4).history()).toEqual(h);
  });
  it("does not mutate the input list", () => {
    const copy = [...h];
    boardAtPly(h, 3);
    expect(h).toEqual(copy);
  });
});

describe("truncateAndPlay", () => {
  const h = ["e4", "e5", "Nf3", "Nc6"];
  it("truncates at ply and appends a legal move's SAN", () => {
    expect(truncateAndPlay(h, 2, { from: "g1", to: "f3" })).toEqual(["e4", "e5", "Nf3"]);
    expect(truncateAndPlay(h, 1, { from: "c7", to: "c5" })).toEqual(["e4", "c5"]); // diverge from the line
  });
  it("appends promotion moves", () => {
    const promo = truncateAndPlay(["e4", "d5", "exd5", "c6", "dxc6", "b6", "cxb7", "Bd7"], 8,
                                  { from: "b7", to: "a8", promotion: "q" });
    expect(promo && promo[promo.length - 1]).toBe("bxa8=Q+");
  });
  it("returns null for an illegal move", () => {
    expect(truncateAndPlay(h, 4, { from: "e1", to: "e5" })).toBeNull();
  });
});

describe("shouldBotReply", () => {
  it("true only when it's the bot's turn and the game is live", () => {
    expect(shouldBotReply(boardAtPly(["e4"], 1), "b")).toBe(true);  // black to move
    expect(shouldBotReply(boardAtPly(["e4"], 1), "w")).toBe(false); // not bot's turn
    const mate = boardAtPly(["f3", "e5", "g4", "Qh4#"], 4);
    expect(shouldBotReply(mate, "w")).toBe(false);                  // game over
  });
});
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding/web && npx vitest run src/gameNav.test.ts'`
Expected: FAIL — cannot resolve `./gameNav`.

- [ ] **Step 3: Create `web/src/gameNav.ts`**

```ts
import { Chess } from "chess.js";

// A fresh board with the first `ply` half-moves of `history` (SAN) applied. Pure: never mutates input.
export function boardAtPly(history: string[], ply: number): Chess {
  const c = new Chess();
  for (let i = 0; i < ply && i < history.length; i++) c.move(history[i]);
  return c;
}

// history[0:ply] + `move`, returned as a new SAN list — or null if the move is illegal there.
export function truncateAndPlay(
  history: string[], ply: number, move: { from: string; to: string; promotion?: string },
): string[] | null {
  const c = boardAtPly(history, ply);
  try {
    const m = c.move({ from: move.from, to: move.to, promotion: move.promotion ?? "q" });
    return [...history.slice(0, ply), m.san];
  } catch {
    return null; // chess.js v1 THROWS on an illegal move
  }
}

// The bot should auto-reply only when the move just played handed it the turn.
export function shouldBotReply(board: Chess, botColor: "w" | "b"): boolean {
  return !board.isGameOver() && board.turn() === botColor;
}
```

- [ ] **Step 4: Run the test**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding/web && npx vitest run src/gameNav.test.ts'`
Expected: PASS (3 describes, all green).

- [ ] **Step 5: Commit**

```bash
git add web/src/gameNav.ts web/src/gameNav.test.ts
git commit -m "feat(web): pure game-navigation helpers (boardAtPly, truncateAndPlay, shouldBotReply)"
```

---

## Task 2: BoardPanel — navigable game model + force-move + single analysis panel

**Files:**
- Rewrite: `web/src/components/BoardPanel.tsx`
- Delete: `web/src/undo.ts`, `web/src/undo.test.ts`

**Interfaces:**
- Consumes: `gameNav` (Task 1); existing `resolveClick`, `topMoves`, `engine.value`, `bookOrModelMove`, `botColorOf`/`boardOrientationOf`/`botShouldOpen`, `WDLBar`.
- Produces: `BoardPanel` with the SAME props as today (`{ engine, elo, temperature, books, playerColor }`) — the elo split happens in Task 3. Internally: SAN-list + `viewPly` model, ◀/▶ + arrow nav, move-for-side-to-move, one analysis panel ("What would play here") + highlight at `elo`, WDL bar at `elo`.

This task changes behavior (single navigable panel, nav buttons replace Undo, force-move enabled) but keeps the prop signature, so it compiles against today's `App`/`Controls`.

- [ ] **Step 1: Delete the obsolete undo module + its test**

```bash
git rm web/src/undo.ts web/src/undo.test.ts
```

- [ ] **Step 2: Rewrite `web/src/components/BoardPanel.tsx`**

Replace the ENTIRE file with:

```tsx
import React, { useCallback, useEffect, useRef, useState } from "react";
import { Chessboard } from "react-chessboard";
import { topMoves } from "../inference/topMoves";
import { bookOrModelMove } from "../inference/bookMove";
import { ThinkingPanel } from "./ThinkingPanel";
import { botColorOf, boardOrientationOf, botShouldOpen } from "../playerColor";
import { WDLBar, type WDL } from "./WDLBar";
import { resolveClick } from "../clickMove";
import { boardAtPly, truncateAndPlay, shouldBotReply } from "../gameNav";
import type { Engine } from "../inference/engine";
import type { OpeningBookSet } from "../inference/openingBook";

const MOVE_DELAY_MS = 650; // brief pause so the bot's reply is easy to follow

type MoveProb = { uci: string; san: string; prob: number };

export function BoardPanel({ engine, elo, temperature, books, playerColor }:
  { engine: Engine | null; elo: number; temperature: number; books: OpeningBookSet | null; playerColor: "w" | "b" }) {
  const botColor = botColorOf(playerColor);

  // The current line as an authoritative SAN list; `viewPly` = how many plies are shown (length = live tip).
  const [history, setHistory] = useState<string[]>([]);
  const historyRef = useRef(history);
  historyRef.current = history; // keep the ref synced so the async bot reply reads the latest line
  const [viewPly, setViewPly] = useState(0);
  const [thinking, setThinking] = useState(false);
  const [analysis, setAnalysis] = useState<MoveProb[]>([]);
  const [wdl, setWdl] = useState<WDL | null>(null);
  const [wdlStm, setWdlStm] = useState<"w" | "b">("w");
  const [selected, setSelected] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const board = boardAtPly(history, viewPly); // the displayed position
  const tip = history.length;
  const atTip = viewPly === tip;

  // Commit a new line: set the ref synchronously (for async reads) and jump the view to its tip.
  const commit = useCallback((next: string[]) => {
    historyRef.current = next;
    setHistory(next);
    setViewPly(next.length);
  }, []);

  const botMove = useCallback(async () => {
    if (!engine) return;
    if (boardAtPly(historyRef.current, historyRef.current.length).isGameOver()) return;
    setThinking(true);
    try {
      await new Promise((r) => setTimeout(r, MOVE_DELAY_MS)); // let the human see their move land first
      const cur = boardAtPly(historyRef.current, historyRef.current.length);
      if (cur.isGameOver()) return;
      const mv = await bookOrModelMove(books, engine, cur, elo, { temperature, greedy: false });
      const next = truncateAndPlay(historyRef.current, historyRef.current.length, mv);
      if (next) commit(next);
    } finally {
      setThinking(false);
    }
  }, [books, engine, elo, temperature, commit]);

  const botMoveRef = useRef(botMove);
  botMoveRef.current = botMove;

  // Apply a move for the side to move at the viewed ply; truncates any later plies (diverging the
  // line). Shared by drag + tap. The bot replies only if the move handed it the turn.
  const playMove = useCallback((from: string, to: string): boolean => {
    if (thinking) return false;
    const next = truncateAndPlay(historyRef.current, viewPly, { from, to });
    if (!next) return false;
    commit(next);
    if (shouldBotReply(boardAtPly(next, next.length), botColor)) void botMoveRef.current();
    return true;
  }, [thinking, viewPly, botColor, commit]);

  const onDrop = useCallback((from: string, to: string) => playMove(from, to), [playMove]);

  // Tap-to-move: tap a piece of the side-to-move to select, tap a target to move.
  const onSquareClick = useCallback((square: string) => {
    if (thinking) return;
    const b = boardAtPly(historyRef.current, viewPly);
    if (b.isGameOver()) return;
    const r = resolveClick(b, selected, square, b.turn()); // mover = the side to move at the viewed ply
    if (r.type === "select") setSelected(r.from);
    else if (r.type === "deselect" || r.type === "ignore") setSelected(null);
    else if (r.type === "move") { if (!playMove(r.from, r.to)) setSelected(null); }
  }, [thinking, viewPly, selected, playMove]);

  const goBack = useCallback(() => { if (!thinking) setViewPly((p) => Math.max(0, p - 1)); }, [thinking]);
  const goForward = useCallback(() => {
    if (!thinking) setViewPly((p) => Math.min(historyRef.current.length, p + 1));
  }, [thinking]);

  // Arrow keys step through history (ignored while a form control is focused).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (document.activeElement as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      if (e.key === "ArrowLeft") goBack();
      else if (e.key === "ArrowRight") goForward();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [goBack, goForward]);

  const newGame = useCallback(() => {
    if (thinking) return;
    commit([]);
    if (playerColor === "b") void botMoveRef.current();
  }, [thinking, playerColor, commit]);

  const copyMoves = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(boardAtPly(historyRef.current, historyRef.current.length).pgn());
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard unavailable (blocked/insecure context) — silently ignore
    }
  }, []);

  // Picking a color starts a fresh game.
  useEffect(() => { commit([]); }, [playerColor, commit]);

  // If the human is Black, the bot (White) opens the fresh game once the engine is ready.
  useEffect(() => {
    if (engine && botShouldOpen(playerColor, historyRef.current.length)) void botMoveRef.current();
  }, [engine, playerColor]);

  // Clear the tap selection on any position/view change.
  useEffect(() => { setSelected(null); }, [history, viewPly]);

  // Analysis panel (top moves for the viewed side-to-move) + WDL bar, recomputed on view/elo change.
  useEffect(() => {
    if (!engine) { setAnalysis([]); setWdl(null); return; }
    const b = boardAtPly(history, viewPly);
    let cancelled = false;
    (async () => {
      if (!b.isGameOver()) {
        const list = await topMoves(engine, b, elo, 5);
        if (!cancelled) setAnalysis(list);
      } else if (!cancelled) setAnalysis([]);
      const stm = b.turn();
      try {
        const v = await engine.value(b, elo);
        if (!cancelled) { setWdl(v); setWdlStm(stm); }
      } catch { if (!cancelled) setWdl(null); }
    })().catch(() => {});
    return () => { cancelled = true; };
  }, [engine, history, viewPly, elo]);

  // Last move into the viewed position (yellow highlight + label).
  const verbose = board.history({ verbose: true });
  const lastMove = verbose.length ? verbose[verbose.length - 1] : null;

  const customSquareStyles: Record<string, React.CSSProperties> = {};
  if (analysis.length > 0) { // top suggestion at the analysis elo
    const top = analysis[0];
    customSquareStyles[top.uci.slice(0, 2)] = { background: "rgba(74,144,217,0.5)" };
    customSquareStyles[top.uci.slice(2, 4)] = { background: "rgba(74,144,217,0.5)" };
  }
  if (lastMove) {
    customSquareStyles[lastMove.from] = { background: "rgba(255,213,79,0.6)" };
    customSquareStyles[lastMove.to] = { background: "rgba(255,213,79,0.6)" };
  }
  if (selected) {
    customSquareStyles[selected] = { ...customSquareStyles[selected], background: "rgba(74,144,217,0.55)" };
    for (const m of board.moves({ square: selected as any, verbose: true })) {
      customSquareStyles[m.to] = {
        ...customSquareStyles[m.to],
        background: (m as any).captured
          ? "radial-gradient(circle, transparent 58%, rgba(74,144,217,0.45) 60%)"
          : "radial-gradient(circle, rgba(74,144,217,0.5) 22%, transparent 24%)",
      };
    }
  }

  const status = thinking ? "Bot is thinking…"
    : !atTip ? `Viewing move ${viewPly}/${tip}`
    : lastMove ? `Last move: ${lastMove.san}` : "";

  return (
    <div style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>
      <WDLBar wdl={wdl} sideToMove={wdlStm} playerColor={playerColor} height={480} />
      <div style={{ width: 480 }}>
        <Chessboard
          position={board.fen()}
          onPieceDrop={onDrop}
          onSquareClick={onSquareClick}
          arePiecesDraggable={!thinking}
          customSquareStyles={customSquareStyles}
          boardWidth={480}
          boardOrientation={boardOrientationOf(playerColor)}
        />
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 8, flexWrap: "wrap", minHeight: 24 }}>
          <button onClick={newGame} disabled={thinking}>New game</button>
          <button onClick={goBack} disabled={thinking || viewPly === 0}>◀</button>
          <button onClick={goForward} disabled={thinking || atTip}>▶</button>
          <button onClick={copyMoves} disabled={tip === 0}>{copied ? "Copied!" : "Copy moves"}</button>
          <span style={{ color: "#555" }}>{status}</span>
        </div>
        {board.isGameOver() && <p>Game over: {board.isCheckmate() ? "checkmate" : "draw"}</p>}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
        <ThinkingPanel title="What would play here" moves={analysis} emptyHint="—" />
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Run the full web suite**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding/web && npx vitest run'`
Expected: PASS — all remaining tests (the deleted `undo.test.ts` is gone; `gameNav`, `clickMove`, `WDLBar`, `playerColor`, engine parity all green).

- [ ] **Step 4: Typecheck + build**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding/web && npx tsc --noEmit && npm run build'`
Expected: tsc clean; build succeeds.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/BoardPanel.tsx
git commit -m "feat(web): navigable game model + ply nav + force-move + single analysis panel

Replaces the gameRef-only board with a SAN-list + viewPly cursor. ◀/▶ and
arrow keys step through history; a move at a rewound ply truncates the line
and continues, with the bot replying only when handed the turn. Removes Undo
(superseded by nav) and collapses the two analysis panels into one for the
viewed position."
```

---

## Task 3: Split bot elo / analysis elo, Show-analysis toggle, lock bot elo

**Files:**
- Modify: `web/src/App.tsx`
- Modify: `web/src/components/Controls.tsx`
- Modify: `web/src/components/BoardPanel.tsx`

**Interfaces:**
- `App` owns `botElo`, `analysisElo`, `showAnalysis`, `playerColor`, `temperature`, `gameStarted`.
- `Controls` props: `{ botElo, setBotElo, botEloLocked, analysisElo, setAnalysisElo, showAnalysis, setShowAnalysis, temperature, setTemperature, playerColor, setPlayerColor }`.
- `BoardPanel` props become `{ engine, botElo, analysisElo, showAnalysis, temperature, books, playerColor, onGameStartedChange }` (replacing `elo`).

- [ ] **Step 1: Rewrite `web/src/App.tsx`**

Replace the entire file with:

```tsx
import React, { useState } from "react";
import { useEngine } from "./useEngine";
import { BoardPanel } from "./components/BoardPanel";
import { Controls } from "./components/Controls";

export function App() {
  const { engine, error, books } = useEngine();
  const [botElo, setBotElo] = useState(1500);
  const [analysisElo, setAnalysisElo] = useState(1500);
  const [showAnalysis, setShowAnalysis] = useState(true);
  const [temperature, setTemperature] = useState(0.1);
  const [playerColor, setPlayerColor] = useState<"w" | "b">("w");
  const [gameStarted, setGameStarted] = useState(false); // locks the bot-elo control once a move is played
  return (
    <div style={{ maxWidth: 760, margin: "0 auto", padding: 16 }}>
      <h1>Eloquent Bot</h1>
      {error && <p style={{ color: "crimson" }}>Failed to load model: {error}</p>}
      {!engine && !error && <p>Loading model…</p>}
      <Controls
        botElo={botElo} setBotElo={setBotElo} botEloLocked={gameStarted}
        analysisElo={analysisElo} setAnalysisElo={setAnalysisElo}
        showAnalysis={showAnalysis} setShowAnalysis={setShowAnalysis}
        temperature={temperature} setTemperature={setTemperature}
        playerColor={playerColor} setPlayerColor={setPlayerColor}
      />
      <BoardPanel
        engine={engine} botElo={botElo} analysisElo={analysisElo} showAnalysis={showAnalysis}
        temperature={temperature} books={books} playerColor={playerColor}
        onGameStartedChange={setGameStarted}
      />
    </div>
  );
}
```

- [ ] **Step 2: Rewrite `web/src/components/Controls.tsx`**

Replace the entire file with:

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
    <div style={{ display: "flex", gap: 24, margin: "12px 0", alignItems: "center", flexWrap: "wrap" }}>
      <label>Bot elo: {botElo}{botEloLocked ? " 🔒" : ""}
        <input type="range" min={600} max={2400} step={100} value={botElo} disabled={botEloLocked}
               onChange={(e) => setBotElo(Number(e.target.value))} />
      </label>
      <label style={{ opacity: showAnalysis ? 1 : 0.5 }}>Analysis elo: {analysisElo}
        <input type="range" min={600} max={2400} step={100} value={analysisElo} disabled={!showAnalysis}
               onChange={(e) => setAnalysisElo(Number(e.target.value))} />
      </label>
      <label>
        <input type="checkbox" checked={showAnalysis} onChange={(e) => setShowAnalysis(e.target.checked)} />
        {" "}Show analysis
      </label>
      <label>Temperature: {temperature.toFixed(1)}
        <input type="range" min={0.1} max={2.0} step={0.1} value={temperature}
               onChange={(e) => setTemperature(Number(e.target.value))} />
      </label>
      <span>
        Play as:{" "}
        <button onClick={() => setPlayerColor("w")} disabled={playerColor === "w"}>White</button>{" "}
        <button onClick={() => setPlayerColor("b")} disabled={playerColor === "b"}>Black</button>
      </span>
    </div>
  );
}
```

- [ ] **Step 3: Update `BoardPanel.tsx` props + elo wiring**

(a) Change the signature + destructure:

```tsx
export function BoardPanel({ engine, botElo, analysisElo, showAnalysis, temperature, books, playerColor, onGameStartedChange }:
  { engine: Engine | null; botElo: number; analysisElo: number; showAnalysis: boolean;
    temperature: number; books: OpeningBookSet | null; playerColor: "w" | "b";
    onGameStartedChange: (started: boolean) => void }) {
```

(b) In `botMove`, change the strength to `botElo` (both the `bookOrModelMove` call and the dep array):

```tsx
      const mv = await bookOrModelMove(books, engine, cur, botElo, { temperature, greedy: false });
```
and its deps: `}, [books, engine, botElo, temperature, commit]);`

(c) Replace the analysis/WDL effect with one that uses `analysisElo` for the panel (gated by `showAnalysis`) and `botElo` for the WDL bar:

```tsx
  // Analysis panel (top moves at the analysis elo, side-to-move) + WDL bar (at the bot's elo).
  useEffect(() => {
    if (!engine) { setAnalysis([]); setWdl(null); return; }
    const b = boardAtPly(history, viewPly);
    let cancelled = false;
    (async () => {
      if (showAnalysis && !b.isGameOver()) {
        const list = await topMoves(engine, b, analysisElo, 5);
        if (!cancelled) setAnalysis(list);
      } else if (!cancelled) setAnalysis([]);
      const stm = b.turn();
      try {
        const v = await engine.value(b, botElo);
        if (!cancelled) { setWdl(v); setWdlStm(stm); }
      } catch { if (!cancelled) setWdl(null); }
    })().catch(() => {});
    return () => { cancelled = true; };
  }, [engine, history, viewPly, analysisElo, botElo, showAnalysis]);
```

(d) Report game-started so the bot-elo control locks once a move exists. Add this effect (next to the other effects):

```tsx
  // The bot elo is locked once any move has been played; report that to the parent control.
  useEffect(() => { onGameStartedChange(history.length > 0); }, [history, onGameStartedChange]);
```

(e) Hide the analysis panel when the toggle is off — wrap the panel column:

```tsx
      {showAnalysis && (
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <ThinkingPanel title="What would play here" moves={analysis} emptyHint="—" />
        </div>
      )}
```

(With `showAnalysis` off, `analysis` is set to `[]` by the effect, so the board's blue highlight also disappears.)

- [ ] **Step 4: Typecheck, build, full suite**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding/web && npx tsc --noEmit && npm run build && npx vitest run'`
Expected: tsc clean; build succeeds; all tests pass (no test imports the old single-`elo` prop).

- [ ] **Step 5: Commit**

```bash
git add web/src/App.tsx web/src/components/Controls.tsx web/src/components/BoardPanel.tsx
git commit -m "feat(web): fixed bot elo (locked at start) + scrub-able analysis elo + show-analysis toggle

Bot plays at botElo (locked once a move is played, re-enabled on New Game); a
separate analysis-elo slider drives the live top-moves panel + board highlight;
a Show-analysis checkbox hides both. WDL bar stays on botElo."
```

---

## Self-review notes

- **Spec coverage:** ply nav + force via truncate-and-continue (T1 helpers + T2); single analysis panel + live highlight at analysis elo (T2 panel, T3 elo); optional suggestion via Show-analysis (T3); fixed/locked bot elo (T3); WDL bar on botElo, analysis on analysisElo, bot move on botElo (T3, per Global Constraints). Pure-helper tests (gameNav) + existing suites; wiring by tsc/build.
- **Naming consistency:** `boardAtPly`/`truncateAndPlay`/`shouldBotReply`, `viewPly`/`history`/`commit`, `botElo`/`analysisElo`/`showAnalysis`/`gameStarted`/`onGameStartedChange`, `botEloLocked` used identically across tasks.
- **Known follow-ups (out of scope):** player-elo estimate (round B); coaching + WDL sparkline (round B candidates); visual redesign + mobile/responsive (round C).
```
