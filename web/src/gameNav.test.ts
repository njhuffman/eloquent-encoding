import { describe, it, expect } from "vitest";
import { Chess } from "chess.js";
import { boardAtPly, truncateAndPlay, shouldBotReply } from "./gameNav";

describe("boardAtPly", () => {
  const h = ["e4", "e5", "Nf3", "Nc6"];
  it("replays the first `ply` moves into a fresh board", () => {
    expect(boardAtPly(h, 0).fen()).toBe(new Chess().fen());
    expect(boardAtPly(h, 2).history()).toEqual(["e4", "e5"]);
    expect(boardAtPly(h, 4).history()).toEqual(h);
  });
  it("does not mutate the input list", () => {
    const copy = [...h];
    boardAtPly(h, 3);
    expect(h).toEqual(copy);
  });
});

describe("truncateAndPlay", () => {
  const h = ["e4", "e5", "Nf3", "Nc6"];
  it("truncates at ply and appends a legal move's SAN", () => {
    expect(truncateAndPlay(h, 2, { from: "g1", to: "f3" })).toEqual(["e4", "e5", "Nf3"]);
    expect(truncateAndPlay(h, 1, { from: "c7", to: "c5" })).toEqual(["e4", "c5"]); // diverge from the line
  });
  it("appends promotion moves", () => {
    const promo = truncateAndPlay(["a4", "d5", "a5", "Kd7", "a6", "Kc6", "axb7", "e6"], 8,
                                  { from: "b7", to: "a8", promotion: "q" });
    expect(promo && promo[promo.length - 1]).toBe("bxa8=Q+");
  });
  it("returns null for an illegal move", () => {
    expect(truncateAndPlay(h, 4, { from: "e1", to: "e5" })).toBeNull();
  });
});

describe("shouldBotReply", () => {
  it("true only when it's the bot's turn and the game is live", () => {
    expect(shouldBotReply(boardAtPly(["e4"], 1), "b")).toBe(true);  // black to move
    expect(shouldBotReply(boardAtPly(["e4"], 1), "w")).toBe(false); // not bot's turn
    const mate = boardAtPly(["f3", "e5", "g4", "Qh4#"], 4);
    expect(shouldBotReply(mate, "w")).toBe(false);                  // game over
  });
});
