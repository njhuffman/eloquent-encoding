import { describe, it, expect } from "vitest";
import { botColorOf, boardOrientationOf, botShouldOpen } from "./playerColor";

describe("player color helpers", () => {
  it("botColorOf is the opposite color", () => {
    expect(botColorOf("w")).toBe("b");
    expect(botColorOf("b")).toBe("w");
  });
  it("boardOrientationOf maps to react-chessboard strings", () => {
    expect(boardOrientationOf("w")).toBe("white");
    expect(boardOrientationOf("b")).toBe("black");
  });
  it("bot opens only when the human is Black and the board is fresh", () => {
    expect(botShouldOpen("b", 0)).toBe(true);
    expect(botShouldOpen("b", 1)).toBe(false);
    expect(botShouldOpen("w", 0)).toBe(false);
  });
});
