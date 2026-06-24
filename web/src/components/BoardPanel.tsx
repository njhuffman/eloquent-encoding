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
