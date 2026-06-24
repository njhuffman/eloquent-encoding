import { describe, it, expect } from "vitest";
import { posteriorFromLogProbs, shouldScorePly } from "./eloEstimate";

const L = Math.log;
const bands = [1000, 1500, 2000];

describe("posteriorFromLogProbs", () => {
  it("equal log-probs → uniform posterior, mean = average band", () => {
    const r = posteriorFromLogProbs([[L(0.3), L(0.3), L(0.3)]], bands);
    expect(r.posterior[0]).toBeCloseTo(1 / 3, 6);
    expect(r.posterior[1]).toBeCloseTo(1 / 3, 6);
    expect(r.posterior[2]).toBeCloseTo(1 / 3, 6);
    expect(r.meanElo).toBeCloseTo(1500, 6);
  });

  it("a move one band loves pulls the posterior + MAP to it", () => {
    const r = posteriorFromLogProbs([[L(0.05), L(0.8), L(0.05)]], bands);
    expect(r.mapElo).toBe(1500);
    expect(r.posterior[1]).toBeGreaterThan(r.posterior[0]);
    expect(r.posterior[1]).toBeGreaterThan(r.posterior[2]);
  });

  it("accumulates across moves", () => {
    const r = posteriorFromLogProbs([[L(0.1), L(0.2), L(0.7)], [L(0.1), L(0.2), L(0.7)]], bands);
    expect(r.mapElo).toBe(2000);
    expect(r.meanElo).toBeGreaterThan(1500);
  });

  it("empty input → uniform posterior", () => {
    const r = posteriorFromLogProbs([], bands);
    for (const p of r.posterior) expect(p).toBeCloseTo(1 / 3, 6);
    expect(r.meanElo).toBeCloseTo(1500, 6);
  });
});

describe("shouldScorePly", () => {
  it("scores the player's own moves only past the skipped opening plies", () => {
    expect(shouldScorePly(4, "w", "w", 4)).toBe(true);  // first in-distribution white ply
    expect(shouldScorePly(3, "w", "w", 4)).toBe(false); // opening ply -> skipped
    expect(shouldScorePly(0, "w", "w", 4)).toBe(false);
  });
  it("excludes the opponent's / bot's moves", () => {
    expect(shouldScorePly(5, "b", "w", 4)).toBe(false); // not the player's color
    expect(shouldScorePly(6, "b", "b", 4)).toBe(true);  // black player, past opening
  });
});
