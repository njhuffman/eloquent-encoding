import type { Chess } from "chess.js";

// Result of tapping a square, given the currently-selected from-square (if any).
// The caller guards the turn/thinking/game-over conditions and validates legality.
export type ClickResult =
  | { type: "select"; from: string }
  | { type: "deselect" }
  | { type: "move"; from: string; to: string }
  | { type: "ignore" };

// Tap-to-move resolution (mobile-friendly alternative to drag):
//  - nothing selected: tapping your own piece selects it; anything else is ignored.
//  - a piece selected: tapping it again deselects; tapping another of your pieces reselects;
//    tapping any other square is a move attempt (caller validates legality).
export function resolveClick(
  board: Chess, selected: string | null, square: string, playerColor: "w" | "b",
): ClickResult {
  const piece = board.get(square as any); // chess.js: Piece | null
  const isOwnPiece = !!piece && piece.color === playerColor;
  if (!selected) {
    return isOwnPiece ? { type: "select", from: square } : { type: "ignore" };
  }
  if (square === selected) return { type: "deselect" };
  if (isOwnPiece) return { type: "select", from: square };
  return { type: "move", from: selected, to: square };
}
