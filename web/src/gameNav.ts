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
