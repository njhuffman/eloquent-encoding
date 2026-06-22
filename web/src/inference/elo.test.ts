import { describe, it, expect } from "vitest";
import { eloToBucket } from "./elo";
describe("eloToBucket", () => {
  it("matches python: floor(elo/100) clamped to [0,n-1], 0 -> null index n", () => {
    expect(eloToBucket(1500, 40)).toBe(15);
    expect(eloToBucket(1200, 40)).toBe(12);
    expect(eloToBucket(50, 40)).toBe(0);
    expect(eloToBucket(9999, 40)).toBe(39);   // clamp
    expect(eloToBucket(0, 40)).toBe(40);       // null bucket
  });
});
