import type { Chess } from "chess.js";
import type { Engine } from "./engine";
import { maskedSoftmax } from "./sample";
import { legalFromMask, legalToMask } from "./legal";
import { indexToSquare } from "./boardTensor";

// Needs raw logits per from-square; expose a helper on Engine via policy-style calls.
export async function topMoves(engine: Engine, board: Chess, elo: number, k: number) {
  const dist = await engine.distributions(board, elo);   // see Engine.distributions below
  const fromMask = legalFromMask(board);
  const fromProbs = maskedSoftmax(dist.fromLogits, fromMask, 1.0);
  const out: { uci: string; san: string; prob: number }[] = [];
  for (let f = 0; f < 64; f++) {
    if (!fromMask[f]) continue;
    const toLogits = await dist.toLogits(f);
    const toProbs = maskedSoftmax(toLogits, legalToMask(board, f), 1.0);
    for (let t = 0; t < 64; t++) {
      if (toProbs[t] <= 0) continue;
      const uci = indexToSquare(f) + indexToSquare(t);
      const probe = new (board.constructor as any)(board.fen());
      const mv = probe.move({ from: indexToSquare(f), to: indexToSquare(t), promotion: "q" });
      if (!mv) continue;
      out.push({ uci, san: mv.san, prob: fromProbs[f] * toProbs[t] });
    }
  }
  out.sort((a, b) => b.prob - a.prob);
  return out.slice(0, k);
}
