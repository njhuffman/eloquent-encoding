import { describe, it, expect } from "vitest";
import { boardSizeFor } from "./boardSize";

describe("boardSizeFor", () => {
  it("caps at the max (default 480)", () => {
    expect(boardSizeFor(640)).toBe(480);
    expect(boardSizeFor(517.8)).toBe(480);
  });
  it("uses the container width (floored) when below the max", () => {
    expect(boardSizeFor(300)).toBe(300);
    expect(boardSizeFor(200.9)).toBe(200);
  });
  it("never returns less than 1 (guards 0 / negative / NaN-ish)", () => {
    expect(boardSizeFor(0)).toBe(1);
    expect(boardSizeFor(-50)).toBe(1);
  });
  it("honors a custom max", () => {
    expect(boardSizeFor(900, 600)).toBe(600);
  });
});
