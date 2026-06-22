import React, { useCallback, useState } from "react";
import { Chessboard } from "react-chessboard";
import { Chess } from "chess.js";
import type { Engine } from "../inference/engine";

export function BoardPanel({ engine, elo, temperature }:
  { engine: Engine | null; elo: number; temperature: number }) {
  const [game, setGame] = useState(new Chess());
  const [thinking, setThinking] = useState(false);

  const botMove = useCallback(async (g: Chess) => {
    if (!engine || g.isGameOver()) return;
    setThinking(true);
    const mv = await engine.chooseMove(g, elo, { temperature, greedy: false });
    g.move(mv);
    setGame(new Chess(g.fen()));
    setThinking(false);
  }, [engine, elo, temperature]);

  const onDrop = useCallback((from: string, to: string) => {
    const g = new Chess(game.fen());
    const res = g.move({ from, to, promotion: "q" });
    if (!res) return false;
    setGame(g);
    void botMove(new Chess(g.fen()));
    return true;
  }, [game, botMove]);

  return (
    <div>
      <Chessboard position={game.fen()} onPieceDrop={onDrop} arePiecesDraggable={!thinking} />
      <button onClick={() => setGame(new Chess())}>New game</button>
      {game.isGameOver() && <p>Game over: {game.isCheckmate() ? "checkmate" : "draw"}</p>}
    </div>
  );
}
