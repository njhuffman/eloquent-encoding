import { describe, it, expect } from "vitest";
import { Chess } from "chess.js";
import { OpeningBook, OpeningBookSet, epdKey } from "./openingBook";
import { bookOrModelMove } from "./bookMove";

const start = epdKey(new Chess());
function setReturning(mv: any) {
  const set = new OpeningBookSet("/b/");
  set.forElo = async () => (mv ? new OpeningBook(100, { [start]: { n: 100, moves: { e2e4: 100 } } }) : null);
  return set;
}

describe("bookOrModelMove", () => {
  it("plays the book move and never calls the model", async () => {
    const engine = { chooseMove: async () => { throw new Error("model used on a book hit"); } } as any;
    const mv = await bookOrModelMove(setReturning(true), engine, new Chess(), 1800, { temperature: 1 });
    expect(mv.from + mv.to).toBe("e2e4");
  });
  it("falls through to the model on a book miss", async () => {
    const engine = { chooseMove: async () => ({ from: "g1", to: "f3" }) } as any;
    const mv = await bookOrModelMove(setReturning(false), engine, new Chess(), 1800, { temperature: 1 });
    expect(mv.from + mv.to).toBe("g1f3");
  });
  it("falls through when books is null", async () => {
    const engine = { chooseMove: async () => ({ from: "g1", to: "f3" }) } as any;
    const mv = await bookOrModelMove(null, engine, new Chess(), 1800, { temperature: 1 });
    expect(mv.from + mv.to).toBe("g1f3");
  });
});
