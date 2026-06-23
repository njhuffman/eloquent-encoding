import type { Chess } from "chess.js";

// The human plays White; the bot plays Black. "Undo" takes the position back to
// the human's previous decision point: undo the bot's reply AND the human's move.
// If the bot hasn't replied yet (e.g. the human's move just ended the game), only
// the single last ply is undone — leaving White (the human) to move again.
export function undoToHumanTurn(g: Chess): void {
  if (g.history().length === 0) return;
  g.undo();
  if (g.history().length > 0 && g.turn() !== "w") g.undo();
}
