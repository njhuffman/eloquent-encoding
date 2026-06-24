import { describe, it, expect } from "vitest";
import { arrangeWDL } from "./WDLBar";

describe("arrangeWDL", () => {
  const wdl = { loss: 0.1, draw: 0.3, win: 0.6 };

  it("white player, white to move: white-win at bottom", () => {
    const a = arrangeWDL(wdl, "w", "w");
    expect(a.bottom).toEqual({ kind: "white", prob: 0.6 });
    expect(a.top).toEqual({ kind: "black", prob: 0.1 });
    expect(a.mid).toEqual({ kind: "draw", prob: 0.3 });
  });

  it("black player, white to move: black-win at bottom", () => {
    const a = arrangeWDL(wdl, "w", "b");
    expect(a.bottom).toEqual({ kind: "black", prob: 0.1 });
    expect(a.top).toEqual({ kind: "white", prob: 0.6 });
  });

  it("applies the side-to-move flip (black to move)", () => {
    // black to move: wdl.win is BLACK's win prob
    const a = arrangeWDL(wdl, "b", "w");
    expect(a.bottom).toEqual({ kind: "white", prob: 0.1 }); // white-win = loss for black mover
    expect(a.top).toEqual({ kind: "black", prob: 0.6 });
  });
});
