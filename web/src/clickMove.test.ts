import { describe, it, expect } from "vitest";
import { Chess } from "chess.js";
import { resolveClick } from "./clickMove";

describe("resolveClick", () => {
  const start = new Chess();

  it("selects your own piece when nothing is selected", () => {
    expect(resolveClick(start, null, "e2", "w")).toEqual({ type: "select", from: "e2" });
  });

  it("ignores empty or opponent squares when nothing is selected", () => {
    expect(resolveClick(start, null, "e4", "w")).toEqual({ type: "ignore" }); // empty
    expect(resolveClick(start, null, "e7", "w")).toEqual({ type: "ignore" }); // opponent
  });

  it("deselects when tapping the selected square again", () => {
    expect(resolveClick(start, "e2", "e2", "w")).toEqual({ type: "deselect" });
  });

  it("reselects when tapping another of your pieces", () => {
    expect(resolveClick(start, "e2", "d2", "w")).toEqual({ type: "select", from: "d2" });
  });

  it("treats any other square as a move attempt", () => {
    expect(resolveClick(start, "e2", "e4", "w")).toEqual({ type: "move", from: "e2", to: "e4" });
  });

  it("respects player color for ownership (Black)", () => {
    expect(resolveClick(start, null, "e7", "b")).toEqual({ type: "select", from: "e7" });
    expect(resolveClick(start, null, "e2", "b")).toEqual({ type: "ignore" });
  });
});
