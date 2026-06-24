import type { Chess } from "chess.js";

export function epdKey(board: Chess): string {
  const [placement, turn, castling] = board.fen().split(" ");
  // ep = destination of a legal en-passant capture, else "-" (matches python-chess epd()).
  const epMove = board.moves({ verbose: true }).find((m) => m.flags.includes("e"));
  const ep = epMove ? epMove.to : "-";
  return `${placement} ${turn} ${castling} ${ep}`;
}

export function eloToBand(elo: number): number {
  return Math.max(1000, Math.min(1900, Math.floor(elo / 100) * 100));
}

type Entry = { n: number; moves: Record<string, number> };
export type BookMove = { from: string; to: string; promotion?: string };

export class OpeningBook {
  constructor(public totalGames: number, public positions: Record<string, Entry>) {}

  lookup(board: Chess, threshold: number, rand: () => number): BookMove | null {
    const e = this.positions[epdKey(board)];
    if (!e || this.totalGames <= 0 || e.n / this.totalGames < threshold) return null;
    const legal: Record<string, BookMove> = {};
    for (const m of board.moves({ verbose: true })) {
      const uci = m.from + m.to + (m.promotion ?? "");
      legal[uci] = m.promotion ? { from: m.from, to: m.to, promotion: m.promotion } : { from: m.from, to: m.to };
    }
    const items = Object.entries(e.moves).filter(([uci]) => uci in legal);
    if (!items.length) return null;
    const total = items.reduce((s, [, c]) => s + c, 0);
    const r = rand() * total;
    let acc = 0;
    for (const [uci, c] of items) {
      acc += c;
      if (r <= acc) return legal[uci];
    }
    return legal[items[items.length - 1][0]];
  }

  // The book's most-played legal moves at this position (conditional frequencies), or null if the
  // position isn't in the book above `threshold` — the SAME gate lookup() uses, so the analysis
  // panel shows the book exactly when the bot would actually play from it.
  topMoves(board: Chess, threshold: number, k: number): { uci: string; san: string; prob: number }[] | null {
    const e = this.positions[epdKey(board)];
    if (!e || this.totalGames <= 0 || e.n / this.totalGames < threshold) return null;
    const rows: { uci: string; san: string; count: number }[] = [];
    for (const m of board.moves({ verbose: true })) {
      const c = e.moves[m.from + m.to + (m.promotion ?? "")];
      if (c) rows.push({ uci: m.from + m.to, san: m.san, count: c });
    }
    if (!rows.length) return null;
    const total = rows.reduce((s, r) => s + r.count, 0);
    rows.sort((a, b) => b.count - a.count);
    return rows.slice(0, k).map((r) => ({ uci: r.uci, san: r.san, prob: r.count / total }));
  }
}

export class OpeningBookSet {
  private cache = new Map<number, Promise<OpeningBook | null>>();
  constructor(private baseUrl: string, private fetchFn: typeof fetch = fetch) {}

  forElo(elo: number): Promise<OpeningBook | null> {
    const band = eloToBand(elo);
    let p = this.cache.get(band);
    if (!p) {
      p = this.fetchFn(`${this.baseUrl}opening_book/band_${band}.json`)
        .then((r) => (r.ok ? r.json() : null))
        .then((d) => (d ? new OpeningBook(d.total_games, d.positions) : null))
        .catch(() => null);
      this.cache.set(band, p);
    }
    return p;
  }
}
