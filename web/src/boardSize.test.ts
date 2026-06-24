import { describe, it, expect } from "vitest";
import { boardSizeFor, fitBoardSize } from "./boardSize";

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

describe("fitBoardSize", () => {
  it("is width-bound when the viewport is tall (portrait phone)", () => {
    expect(fitBoardSize(332, 844)).toBe(332); // min(332, 844-160=684) -> 332
  });
  it("is height-bound when the viewport is short (landscape phone)", () => {
    expect(fitBoardSize(609, 375)).toBe(215); // min(609, 375-160=215) -> 215
  });
  it("still caps at the max on a roomy desktop", () => {
    expect(fitBoardSize(700, 1000)).toBe(480); // min(700, 840) -> 480
  });
  it("stays >= 1 even when the reserve exceeds the viewport", () => {
    expect(fitBoardSize(300, 100)).toBe(1); // min(300, -60) -> boardSizeFor(-60) -> 1
  });
});
