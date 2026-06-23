import React, { useCallback, useEffect, useState } from "react";
import { Chessboard } from "react-chessboard";
import { Chess } from "chess.js";
import type { Engine } from "../inference/engine";
import { topMoves } from "../inference/topMoves";
import { ThinkingPanel } from "./ThinkingPanel";

const MOVE_DELAY_MS = 650; // brief pause so the bot's reply is easy to follow

type LastMove = { san: string; from: string; to: string };

export function BoardPanel({ engine, elo, temperature }:
  { engine: Engine | null; elo: number; temperature: number }) {
  const [game, setGame] = useState(new Chess());
  const [thinking, setThinking] = useState(false);
  const [lastMove, setLastMove] = useState<LastMove | null>(null);
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
      await new Promise((r) => setTimeout(r, MOVE_DELAY_MS)); // let the human see their move land first
      const mv = await engine.chooseMove(g, elo, { temperature, greedy: false });
      if (g.isGameOver()) return;
      const applied = g.move(mv);
      setGame(new Chess(g.fen()));
      setLastMove({ san: applied.san, from: applied.from, to: applied.to });
    } finally {
      setThinking(false);
    }
  }, [engine, elo, temperature]);

  const onDrop = useCallback((from: string, to: string) => {
    const g = new Chess(game.fen());
    let applied;
    try {
      applied = g.move({ from, to, promotion: "q" }); // chess.js v1 THROWS on an illegal move (doesn't return null)
    } catch {
      return false; // reject the drag; react-chessboard snaps the piece back
    }
    setGame(g);
    setLastMove({ san: applied.san, from: applied.from, to: applied.to });
    void botMove(new Chess(g.fen()));
    return true;
  }, [game, botMove]);

  const newGame = useCallback(() => {
    setGame(new Chess());
    setLastMove(null);
    setTopMovesList([]);
  }, []);

  // Square highlights: blue = model's current top suggestion, yellow = the last move played.
  const customSquareStyles: Record<string, React.CSSProperties> = {};
  if (topMovesList.length > 0) {
    const top = topMovesList[0];
    customSquareStyles[top.uci.slice(0, 2)] = { background: "rgba(74,144,217,0.5)" };
    customSquareStyles[top.uci.slice(2, 4)] = { background: "rgba(74,144,217,0.5)" };
  }
  if (lastMove) {
    // last-move highlight wins over the suggestion tint on shared squares
    customSquareStyles[lastMove.from] = { background: "rgba(255,213,79,0.6)" };
    customSquareStyles[lastMove.to] = { background: "rgba(255,213,79,0.6)" };
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
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 8, minHeight: 24 }}>
          <button onClick={newGame}>New game</button>
          <span style={{ color: "#555" }}>
            {thinking ? "Bot is thinking…" : lastMove ? `Last move: ${lastMove.san}` : ""}
          </span>
        </div>
        {game.isGameOver() && <p>Game over: {game.isCheckmate() ? "checkmate" : "draw"}</p>}
      </div>
      <ThinkingPanel moves={topMovesList} />
    </div>
  );
}
