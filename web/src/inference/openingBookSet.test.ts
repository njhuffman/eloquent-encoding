import { describe, it, expect, vi } from "vitest";
import { OpeningBookSet, epdKey } from "./openingBook";
import { Chess } from "chess.js";

function fakeFetch(payload: any, ok = true) {
  return vi.fn(async (_url: string) => ({ ok, json: async () => payload } as Response));
}

describe("OpeningBookSet", () => {
  it("maps elo->band, loads + caches the band file", async () => {
    const start = epdKey(new Chess());
    const f = fakeFetch({ band: 1800, total_games: 100, positions: { [start]: { n: 100, moves: { e2e4: 100 } } } });
    const set = new OpeningBookSet("/base/", f as any);
    const a = await set.forElo(1850);
    const b = await set.forElo(1820);            // same band -> cached, no second fetch
    expect(a).not.toBeNull();
    expect(a!.totalGames).toBe(100);
    expect(f).toHaveBeenCalledTimes(1);
    expect(f).toHaveBeenCalledWith("/base/opening_book/band_1800.json");
    expect(b).toBe(a);
  });
  it("returns null when the band file is missing", async () => {
    const set = new OpeningBookSet("/base/", fakeFetch(null, false) as any);
    expect(await set.forElo(1850)).toBeNull();
  });
});
