import React, { useCallback, useEffect, useRef, useState } from "react";
import { Chessboard } from "react-chessboard";
import { Chess } from "chess.js";
import type { Engine } from "../inference/engine";
import type { OpeningBookSet } from "../inference/openingBook";
import { topMoves } from "../inference/topMoves";
import { bookOrModelMove } from "../inference/bookMove";
import { undoToHumanTurn } from "../undo";
import { ThinkingPanel } from "./ThinkingPanel";

const MOVE_DELAY_MS = 650; // brief pause so the bot's reply is easy to follow

type LastMove = { san: string; from: string; to: string };
type MoveProb = { uci: string; san: string; prob: number };
type BotAnalysis = { list: MoveProb[]; chosenUci: string };

export function BoardPanel({ engine, elo, temperature, books }:
  { engine: Engine | null; elo: number; temperature: number; books: OpeningBookSet | null }) {
  // gameRef is the authoritative game (keeps full move history for undo + PGN).
  // `fen` mirrors it in state to drive re-renders.
  const gameRef = useRef(new Chess());
  const [fen, setFen] = useState(gameRef.current.fen());
  const [thinking, setThinking] = useState(false);
  const [lastMove, setLastMove] = useState<LastMove | null>(null);
  const [yourMoves, setYourMoves] = useState<MoveProb[]>([]);     // your options for the current position
  const [botAnalysis, setBotAnalysis] = useState<BotAnalysis | null>(null); // bot's choice at its last move
  const [copied, setCopied] = useState(false);

  // Push gameRef state into render state (fen + last-move label).
  const sync = useCallback(() => {
    const g = gameRef.current;
    setFen(g.fen());
    const h = g.history({ verbose: true });
    const last = h[h.length - 1];
    setLastMove(last ? { san: last.san, from: last.from, to: last.to } : null);
  }, []);

  // Recompute BOTH analyses whenever the position, elo, or engine changes — so they
  // persist through the bot's reply and refresh correctly after an undo.
  //  - "Your move": the model's options for the current (human, White) position.
  //  - "Bot's last move": the model's options at the position before the bot's most
  //    recent Black move, with the move it actually played marked.
  useEffect(() => {
    if (!engine) { setYourMoves([]); setBotAnalysis(null); return; }
    const g = gameRef.current;
    let cancelled = false;
    (async () => {
      // Your move (only meaningful when it's White/human to move and the game is live)
      if (!g.isGameOver() && g.turn() === "w") {
        const ym = await topMoves(engine, new Chess(g.fen()), elo, 5);
        if (!cancelled) setYourMoves(ym);
      } else if (!cancelled) {
        setYourMoves([]);
      }
      // Bot's last move: reconstruct the position just before the last Black move
      const verbose = g.history({ verbose: true });
      let lastBlackIdx = -1;
      for (let i = verbose.length - 1; i >= 0; i--) { if (verbose[i].color === "b") { lastBlackIdx = i; break; } }
      if (lastBlackIdx >= 0) {
        const pre = new Chess();
        for (let i = 0; i < lastBlackIdx; i++) pre.move(verbose[i].san);
        const list = await topMoves(engine, pre, elo, 5);
        const mv = verbose[lastBlackIdx];
        if (!cancelled) setBotAnalysis({ list, chosenUci: mv.from + mv.to });
      } else if (!cancelled) {
        setBotAnalysis(null);
      }
    })().catch(() => {});
    return () => { cancelled = true; };
  }, [engine, fen, elo]);

  const botMove = useCallback(async () => {
    const g = gameRef.current;
    if (!engine || g.isGameOver()) return;
    setThinking(true);
    try {
      await new Promise((r) => setTimeout(r, MOVE_DELAY_MS)); // let the human see their move land first
      const mv = await bookOrModelMove(books, engine, new Chess(g.fen()), elo, { temperature, greedy: false });
      if (g.isGameOver()) return;
      g.move(mv);
      sync();
    } finally {
      setThinking(false);
    }
  }, [books, engine, elo, temperature, sync]);

  const onDrop = useCallback((from: string, to: string) => {
    if (thinking) return false;
    const g = gameRef.current;
    try {
      g.move({ from, to, promotion: "q" }); // chess.js v1 THROWS on an illegal move (doesn't return null)
    } catch {
      return false; // reject the drag; react-chessboard snaps the piece back
    }
    sync();
    void botMove();
    return true;
  }, [thinking, botMove, sync]);

  const undo = useCallback(() => {
    if (thinking) return;
    undoToHumanTurn(gameRef.current); // back to the human's previous turn (their move + the bot's reply)
    sync();
  }, [thinking, sync]);

  const newGame = useCallback(() => {
    if (thinking) return;
    gameRef.current = new Chess();
    setLastMove(null);
    setFen(gameRef.current.fen());
  }, [thinking]);

  const copyMoves = useCallback(async () => {
    const pgn = gameRef.current.pgn();
    try {
      await navigator.clipboard.writeText(pgn);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard unavailable (blocked/insecure context) — silently ignore
    }
  }, []);

  const view = gameRef.current; // in sync with `fen` (every mutation calls sync())
  const hasMoves = view.history().length > 0;

  // Square highlights: blue = your current top suggestion, yellow = the last move played.
  const customSquareStyles: Record<string, React.CSSProperties> = {};
  if (yourMoves.length > 0) {
    const top = yourMoves[0];
    customSquareStyles[top.uci.slice(0, 2)] = { background: "rgba(74,144,217,0.5)" };
    customSquareStyles[top.uci.slice(2, 4)] = { background: "rgba(74,144,217,0.5)" };
  }
  if (lastMove) {
    customSquareStyles[lastMove.from] = { background: "rgba(255,213,79,0.6)" };
    customSquareStyles[lastMove.to] = { background: "rgba(255,213,79,0.6)" };
  }

  return (
    <div style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>
      <div style={{ width: 480 }}>
        <Chessboard
          position={fen}
          onPieceDrop={onDrop}
          arePiecesDraggable={!thinking}
          customSquareStyles={customSquareStyles}
          boardWidth={480}
        />
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 8, flexWrap: "wrap", minHeight: 24 }}>
          <button onClick={newGame} disabled={thinking}>New game</button>
          <button onClick={undo} disabled={thinking || !hasMoves}>Undo</button>
          <button onClick={copyMoves} disabled={!hasMoves}>{copied ? "Copied!" : "Copy moves"}</button>
          <span style={{ color: "#555" }}>
            {thinking ? "Bot is thinking…" : lastMove ? `Last move: ${lastMove.san}` : ""}
          </span>
        </div>
        {view.isGameOver() && <p>Game over: {view.isCheckmate() ? "checkmate" : "draw"}</p>}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
        <ThinkingPanel
          title="Bot's last move"
          moves={botAnalysis?.list ?? []}
          highlightUci={botAnalysis?.chosenUci}
          emptyHint="No bot move yet"
        />
        <ThinkingPanel title="Your move" moves={yourMoves} emptyHint="—" />
      </div>
    </div>
  );
}
