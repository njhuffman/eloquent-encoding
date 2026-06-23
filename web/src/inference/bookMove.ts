import type { Chess } from "chess.js";
import type { Engine } from "./engine";
import type { OpeningBookSet, BookMove } from "./openingBook";

export const BOOK_THRESHOLD = 0.01;

export async function bookOrModelMove(
  books: OpeningBookSet | null, engine: Engine, board: Chess, elo: number,
  opts: { temperature: number; greedy?: boolean; rand?: () => number }, threshold = BOOK_THRESHOLD,
): Promise<BookMove> {
  if (books) {
    const bk = await books.forElo(elo);
    const mv = bk?.lookup(board, threshold, Math.random);
    if (mv) return mv;
  }
  return engine.chooseMove(board, elo, opts);
}
