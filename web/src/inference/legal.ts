import type { Chess } from "chess.js";
import { squareToIndex } from "./boardTensor";

export function legalFromMask(board: Chess): boolean[] {
  const m = new Array(64).fill(false);
  for (const mv of board.moves({ verbose: true })) m[squareToIndex(mv.from)] = true;
  return m;
}
export function legalToMask(board: Chess, fromSq: number): boolean[] {
  const m = new Array(64).fill(false);
  for (const mv of board.moves({ verbose: true })) {
    if (squareToIndex(mv.from) === fromSq) m[squareToIndex(mv.to)] = true;
  }
  return m;
}
