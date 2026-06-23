import React, { useCallback, useEffect, useRef, useState } from "react";
import { Chessboard } from "react-chessboard";
import { Chess } from "chess.js";
import type { Engine } from "../inference/engine";
import { topMoves } from "../inference/topMoves";
import { undoToHumanTurn } from "../undo";
import { ThinkingPanel } from "./ThinkingPanel";

const MOVE_DELAY_MS = 650; // brief pause so the bot's reply is easy to follow

type LastMove = { san: string; from: string; to: string };

export function BoardPanel({ engine, elo, temperature }:
  { engine: Engine | null; elo: number; temperature: number }) {
  // gameRef is the authoritative game (keeps full move history for undo + PGN).
  // `fen` mirrors it in state to drive re-renders.
  const gameRef = useRef(new Chess());
  const [fen, setFen] = useState(gameRef.current.fen());
  const [thinking, setThinking] = useState(false);
  const [lastMove, setLastMove] = useState<LastMove | null>(null);
  const [topMovesList, setTopMovesList] = useState<{ uci: string; san: string; prob: number }[]>([]);
  const [copied, setCopied] = useState(false);

  // Push gameRef state into render state (fen + last-move label).
  const sync = useCallback(() => {
    const g = gameRef.current;
    setFen(g.fen());
    const h = g.history({ verbose: true });
    const last = h[h.length - 1];
    setLastMove(last ? { san: last.san, from: last.from, to: last.to } : null);
  }, []);

  // Recompute top moves whenever the position, elo, or engine changes.
  useEffect(() => {
    const board = new Chess(fen);
    if (!engine || board.isGameOver()) { setTopMovesList([]); return; }
    let cancelled = false;
    topMoves(engine, board, elo, 5).then((moves) => {
      if (!cancelled) setTopMovesList(moves);
    }).catch(() => {});
    return () => { cancelled = true; };
  }, [engine, fen, elo]);

  const botMove = useCallback(async () => {
    const g = gameRef.current;
    if (!engine || g.isGameOver()) return;
    setThinking(true);
    try {
      await new Promise((r) => setTimeout(r, MOVE_DELAY_MS)); // let the human see their move land first
      const mv = await engine.chooseMove(new Chess(g.fen()), elo, { temperature, greedy: false });
      if (g.isGameOver()) return;
      g.move(mv);
      sync();
    } finally {
      setThinking(false);
    }
  }, [engine, elo, temperature, sync]);

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
    setTopMovesList([]);
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

  // Square highlights: blue = model's current top suggestion, yellow = the last move played.
  const customSquareStyles: Record<string, React.CSSProperties> = {};
  if (topMovesList.length > 0) {
    const top = topMovesList[0];
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
      <ThinkingPanel moves={topMovesList} />
    </div>
  );
}
