import { describe, it, expect } from "vitest";
import { Chess } from "chess.js";
import { epdKey, eloToBand, OpeningBook } from "./openingBook";
import { mulberry32 } from "./rng";
import cases from "./__fixtures__/epd_cases.json";

describe("epdKey parity vs python-chess epd()", () => {
  it("matches the stored epd for every fixture (ep included)", () => {
    for (const c of cases as { fen: string; epd: string }[]) {
      expect(epdKey(new Chess(c.fen))).toBe(c.epd);
    }
  });
});

describe("eloToBand", () => {
  it("clamps to [1000,1900]", () => {
    expect(eloToBand(1850)).toBe(1800);
    expect(eloToBand(600)).toBe(1000);
    expect(eloToBand(2400)).toBe(1900);
  });
});

describe("OpeningBook.lookup", () => {
  const start = epdKey(new Chess());
  const bk = () => new OpeningBook(1000, { [start]: { n: 900, moves: { e2e4: 600, d2d4: 300 } } });

  it("returns a legal book move above threshold, ∝ counts", () => {
    const n_e4 = Array.from({ length: 200 }, (_, s) =>
      bk().lookup(new Chess(), 0.01, mulberry32(s))).filter((m) => m && m.from === "e2" && m.to === "e4").length;
    expect(n_e4).toBeGreaterThan(120);  // ~2/3 expected
  });
  it("null below threshold / unknown position", () => {
    expect(new OpeningBook(1000, { [start]: { n: 5, moves: { e2e4: 5 } } })
      .lookup(new Chess(), 0.01, mulberry32(0))).toBeNull();
    expect(new OpeningBook(1000, {}).lookup(new Chess(), 0.01, mulberry32(0))).toBeNull();
  });
  it("only returns legal moves", () => {
    const mv = bk().lookup(new Chess(), 0.01, mulberry32(1));
    expect(new Chess().moves({ verbose: true }).some((m) => m.from === mv!.from && m.to === mv!.to)).toBe(true);
  });
});

describe("OpeningBook.topMoves", () => {
  const start = epdKey(new Chess());
  const bk = () => new OpeningBook(1000, { [start]: { n: 900, moves: { e2e4: 600, d2d4: 300 } } });

  it("ranks the book's legal moves by conditional frequency (SAN)", () => {
    const top = bk().topMoves(new Chess(), 0.01, 5);
    expect(top).not.toBeNull();
    expect(top!.map((m) => m.san)).toEqual(["e4", "d4"]);
    expect(top![0]).toMatchObject({ uci: "e2e4" });
    expect(top![0].prob).toBeCloseTo(600 / 900, 6);
    expect(top![1].prob).toBeCloseTo(300 / 900, 6);
  });
  it("null below threshold / unknown position", () => {
    expect(new OpeningBook(1000, { [start]: { n: 5, moves: { e2e4: 5 } } })
      .topMoves(new Chess(), 0.01, 5)).toBeNull();
    expect(new OpeningBook(1000, {}).topMoves(new Chess(), 0.01, 5)).toBeNull();
  });
});
