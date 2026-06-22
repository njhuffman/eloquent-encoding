import { describe, it, expect } from "vitest";
import { maskedSoftmax, pickIndex } from "./sample";
import { mulberry32 } from "./rng";

describe("maskedSoftmax / pickIndex", () => {
  it("zeros illegal, sums to 1 over legal", () => {
    const legal = [true, false, true, false];
    const p = maskedSoftmax([2, 9, 1, 9], legal, 1.0);
    expect(p[1]).toBe(0); expect(p[3]).toBe(0);
    expect(p[0] + p[2]).toBeCloseTo(1, 6);
    expect(p[0]).toBeGreaterThan(p[2]);
  });
  it("greedy picks the max-prob legal index", () => {
    const p = maskedSoftmax([2, 9, 1, 9], [true, false, true, false], 1.0);
    expect(pickIndex(p, { greedy: true })).toBe(0);
  });
  it("seeded sampling is deterministic", () => {
    const p = maskedSoftmax([1, 1, 1, 1], [true, true, true, true], 1.0);
    const a = pickIndex(p, { greedy: false, rand: mulberry32(42) });
    const b = pickIndex(p, { greedy: false, rand: mulberry32(42) });
    expect(a).toBe(b);
  });
});
