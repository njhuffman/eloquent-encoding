// Posterior over elo bands from per-move log-probabilities (uniform prior over bands).
// logProbsPerMove[i][b] = log P(move i | band b). Returns the normalized posterior, the
// posterior-weighted mean elo, and the argmax (MAP) band.
export function posteriorFromLogProbs(
  logProbsPerMove: number[][], elos: number[],
): { posterior: number[]; meanElo: number; mapElo: number } {
  const n = elos.length;
  const logL = new Array(n).fill(0);
  for (const row of logProbsPerMove) {
    for (let b = 0; b < n; b++) logL[b] += row[b];
  }
  const m = Math.max(...logL);
  const w = logL.map((x) => Math.exp(x - m)); // softmax (max-subtracted for stability)
  const s = w.reduce((a, b) => a + b, 0);
  const posterior = w.map((x) => x / s);
  let mapElo = elos[0], best = -Infinity, meanElo = 0;
  for (let b = 0; b < n; b++) {
    meanElo += posterior[b] * elos[b];
    if (posterior[b] > best) { best = posterior[b]; mapElo = elos[b]; }
  }
  return { posterior, meanElo, mapElo };
}

// Whether a ply should feed the player-elo estimate: it must be the player's own move AND past the
// opening plies the model skipped during training (positions before then are out-of-distribution,
// so their per-band likelihoods are noise). `plyIndex` is 0-based; `skipOpeningPlies` mirrors the
// dataset recipe's `skip_opening_plies` (a move at ply i was a training example iff i >= skip).
export function shouldScorePly(
  plyIndex: number, moveColor: "w" | "b", playerColor: "w" | "b", skipOpeningPlies: number,
): boolean {
  return moveColor === playerColor && plyIndex >= skipOpeningPlies;
}
