import { describe, it, expect } from "vitest";
import { Chess } from "chess.js";
import { legalFromMask, legalToMask } from "./legal";
import { squareToIndex } from "./boardTensor";
import fixtures from "./__fixtures__/cases.json";

describe("legal masks", () => {
  it("matches python legal_from / legal_to per fixture", () => {
    for (const c of fixtures.cases) {
      const board = new Chess(c.fen);
      expect(legalFromMask(board)).toEqual(c.legal_from);
      expect(legalToMask(board, c.to_from_sq)).toEqual(c.legal_to);
    }
  });
});
