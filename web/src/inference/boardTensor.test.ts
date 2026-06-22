import { describe, it, expect } from "vitest";
import { Chess } from "chess.js";
import { boardToTensor, squareToIndex } from "./boardTensor";
import fixtures from "./__fixtures__/cases.json";

describe("boardToTensor", () => {
  it("matches python packed_to_board_tensor for every fixture", () => {
    for (const c of fixtures.cases) {
      const t = boardToTensor(new Chess(c.fen));
      expect(t.length).toBe(c.board_tensor.length);
      for (let i = 0; i < t.length; i++) expect(t[i]).toBeCloseTo(c.board_tensor[i], 5);
    }
  });
  it("indexes squares a1=0, h8=63", () => {
    expect(squareToIndex("a1")).toBe(0);
    expect(squareToIndex("h8")).toBe(63);
  });
});
