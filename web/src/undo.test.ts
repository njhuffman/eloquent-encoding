import { describe, it, expect } from "vitest";
import { Chess } from "chess.js";
import { undoToHumanTurn } from "./undo";

describe("undoToHumanTurn", () => {
  it("undoes the bot's reply and the human's move, back to White to move", () => {
    const g = new Chess();
    g.move("e4"); g.move("e5"); g.move("Nf3"); g.move("Nc6"); // W, B, W, B
    undoToHumanTurn(g);
    expect(g.turn()).toBe("w");
    expect(g.history()).toEqual(["e4", "e5"]); // back to before White's 2nd move
  });

  it("undoes a single ply when the human just ended the game (bot never replied)", () => {
    const g = new Chess();
    g.move("e4"); g.move("e5"); g.move("Qh5"); g.move("Nc6"); g.move("Bc4"); g.move("Nf6"); g.move("Qxf7#");
    expect(g.isCheckmate()).toBe(true);
    expect(g.turn()).toBe("b"); // Black to move but mated — bot never got a turn
    undoToHumanTurn(g);
    expect(g.turn()).toBe("w"); // White (human) to re-try
    expect(g.history().length).toBe(6); // only the mating ply removed
  });

  it("undoes to the start when only one move pair has been played", () => {
    const g = new Chess();
    g.move("e4"); g.move("e5");
    undoToHumanTurn(g);
    expect(g.history()).toEqual([]);
    expect(g.turn()).toBe("w");
  });

  it("does nothing on an empty game", () => {
    const g = new Chess();
    undoToHumanTurn(g);
    expect(g.history()).toEqual([]);
  });
});
