import React, { useCallback, useEffect, useState } from "react";
import { Chessboard } from "react-chessboard";
import { Chess } from "chess.js";
import type { Engine } from "../inference/engine";
import { topMoves } from "../inference/topMoves";
import { ThinkingPanel } from "./ThinkingPanel";

export function BoardPanel({ engine, elo, temperature }:
  { engine: Engine | null; elo: number; temperature: number }) {
  const [game, setGame] = useState(new Chess());
  const [thinking, setThinking] = useState(false);
  const [topMovesList, setTopMovesList] = useState<{ uci: string; san: string; prob: number }[]>([]);

  // Recompute top moves whenever position, elo, or engine changes
  useEffect(() => {
    if (!engine || game.isGameOver()) { setTopMovesList([]); return; }
    let cancelled = false;
    topMoves(engine, game, elo, 5).then((moves) => {
      if (!cancelled) setTopMovesList(moves);
    }).catch(() => {});
    return () => { cancelled = true; };
  }, [engine, game, elo]);

  const botMove = useCallback(async (g: Chess) => {
    if (!engine || g.isGameOver()) return;
    setThinking(true);
    try {
      const mv = await engine.chooseMove(g, elo, { temperature, greedy: false });
      if (g.isGameOver()) return;
      g.move(mv);
      setGame(new Chess(g.fen()));
    } finally {
      setThinking(false);
    }
  }, [engine, elo, temperature]);

  const onDrop = useCallback((from: string, to: string) => {
    const g = new Chess(game.fen());
    try {
      g.move({ from, to, promotion: "q" }); // chess.js v1 THROWS on an illegal move (doesn't return null)
    } catch {
      return false; // reject the drag; react-chessboard snaps the piece back
    }
    setGame(g);
    void botMove(new Chess(g.fen()));
    return true;
  }, [game, botMove]);

  // Highlight the top move's from/to squares
  const customSquareStyles: Record<string, React.CSSProperties> = {};
  if (topMovesList.length > 0) {
    const top = topMovesList[0];
    const fromSq = top.uci.slice(0, 2);
    const toSq = top.uci.slice(2, 4);
    customSquareStyles[fromSq] = { background: "rgba(74,144,217,0.5)" };
    customSquareStyles[toSq] = { background: "rgba(74,144,217,0.5)" };
  }

  return (
    <div style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>
      <div style={{ width: 480 }}>
        <Chessboard
          position={game.fen()}
          onPieceDrop={onDrop}
          arePiecesDraggable={!thinking}
          customSquareStyles={customSquareStyles}
          boardWidth={480}
        />
        <button onClick={() => setGame(new Chess())}>New game</button>
        {game.isGameOver() && <p>Game over: {game.isCheckmate() ? "checkmate" : "draw"}</p>}
      </div>
      <ThinkingPanel moves={topMovesList} />
    </div>
  );
}
