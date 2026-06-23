# Web Opening-Book Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the GitHub Pages bot play the per-elo-band opening book (built in Python) before falling through to the ONNX model.

**Architecture:** Ship the band JSONs as static assets; a TS `OpeningBook` reproduces the python-chess `epd()` key in chess.js (with ep normalized to the legal-ep-capture target) and samples a move ∝ human counts; the bot consults it before the model. A parity fixture is the gating test that the keys match byte-for-byte.

**Tech Stack:** TypeScript, chess.js, Vitest (web); python-chess (fixture generation).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-06-23-web-opening-book-integration-design.md`.
- **Position key = python-chess `board.epd()`** = `"<placement> <turn> <castling> <ep>"`. Keep the existing book keys (no re-key). The TS `epdKey` MUST reproduce it byte-for-byte.
- **ep field rule (the parity crux):** `ep` = the destination square of a legal en-passant capture if one exists, else `"-"`. In TS: `board.moves({verbose:true}).find(m => m.flags.includes("e"))?.to ?? "-"`. (Matches python-chess's "only-when-legal" convention; recomputing from legal moves is robust to chess.js's raw FEN ep behavior.)
- **Book move encoding = UCI** (`from+to+promotion-letter`, e.g. `e2e4`, `e7e8q`) — as written by python `move.uci()`.
- **elo→band** = `clamp(Math.floor(elo/100)*100, 1000, 1900)` (mirrors Python `elo_to_band`).
- **Default threshold = 0.01** (1%); book consulted before the model; book off (returns null) if the band file is absent. Sampling uses an injected `rand: () => number` (default `Math.random`).
- Band files committed at `web/public/opening_book/band_<band>.json`; fetched lazily per band from `import.meta.env.BASE_URL`.
- Web tests run in the container: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding/web && npx vitest run <path>'`. Python: `pytest` (NOT `python -m pytest`); scripts run via `python -m scripts.<name>`.
- Band JSON schema (from `BookBuilder.save_all`): `{"band", "total_games", "positions": {epd: {"n", "moves": {uci: count}}}}`.

---

## File Structure

- `scripts/gen_epd_fixtures.py` (create) — emit `{epd, fen}` parity cases (incl. ep) from python-chess.
- `web/public/opening_book/band_<band>.json` (add, committed) — the 10 band books.
- `web/src/inference/__fixtures__/epd_cases.json` (generated, committed) — parity oracle.
- `web/src/inference/openingBook.ts` (create) — `epdKey`, `eloToBand`, `OpeningBook`, `OpeningBookSet`.
- `web/src/inference/bookMove.ts` (create) — `bookOrModelMove` decision helper (book-first).
- `web/src/useEngine.ts` (modify) — construct + expose an `OpeningBookSet`.
- `web/src/components/BoardPanel.tsx` (modify) — `botMove` uses `bookOrModelMove`.
- Tests under `web/src/inference/`.

---

### Task 1: Parity fixtures + ship the band books

**Files:**
- Create: `scripts/gen_epd_fixtures.py`
- Create (generated, committed): `web/src/inference/__fixtures__/epd_cases.json`
- Add (committed): `web/public/opening_book/band_<band>.json` (×10, copied from the data dir)
- Test: `tests/dataset_generation/test_gen_epd_fixtures.py`

**Interfaces:**
- Produces `build_epd_cases() -> list[dict]` returning `[{"fen": <full FEN>, "epd": <board.epd()>}]`
  for a fixed FEN set including: start position, a **legal-ep** position (epd carries the ep square),
  a **spurious-ep** position (FEN has an ep square but no legal ep capture → epd has `-`), a no-ep
  midgame position, and a partial-castling position.

- [ ] **Step 1: Write the failing test**

`tests/dataset_generation/test_gen_epd_fixtures.py`:

```python
from scripts.gen_epd_fixtures import build_epd_cases

def test_cases_include_legal_and_spurious_ep():
    cases = build_epd_cases()
    by_fen = {c["fen"]: c["epd"] for c in cases}
    # legal ep: 1.e4 e6 2.e5 d5 -> exd6 legal -> epd ends with the ep target d6
    legal = "rnbqkbnr/ppp2ppp/4p3/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
    assert by_fen[legal].split()[-1] == "d6"
    # spurious ep: FEN claims e3 but no black pawn can capture -> python epd drops it to "-"
    spurious = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    assert by_fen[spurious].split()[-1] == "-"
    # every case: epd == python-chess board.epd() of the fen, and has 4 space-separated fields
    import chess
    for c in cases:
        assert c["epd"] == chess.Board(c["fen"]).epd()
        assert len(c["epd"].split()) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 pytest tests/dataset_generation/test_gen_epd_fixtures.py -q'`
Expected: FAIL — `No module named 'scripts.gen_epd_fixtures'`.

- [ ] **Step 3: Implement**

`scripts/gen_epd_fixtures.py`:

```python
#!/usr/bin/env python3
"""Emit {epd, fen} parity cases (python-chess) for the TS opening-book key test."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import chess

_FENS = [
    chess.STARTING_FEN,
    "rnbqkbnr/ppp2ppp/4p3/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",   # legal e.p. (exd6)
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",     # spurious e.p. (no legal capture)
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",  # no e.p. midgame
    "r3k2r/8/8/8/8/8/8/R3K2R b Kq - 0 1",                              # partial castling
]


def build_epd_cases() -> list[dict]:
    # Normalize the fen through python-chess so the stored fen matches its epd convention,
    # then keep the ORIGINAL fen string too so chess.js loads the same position.
    out = []
    for fen in _FENS:
        b = chess.Board(fen)
        out.append({"fen": fen, "epd": b.epd()})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="web/src/inference/__fixtures__/epd_cases.json")
    args = ap.parse_args()
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(build_epd_cases(), indent=2))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test, generate the fixture, copy the band books**

```bash
docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 pytest tests/dataset_generation/test_gen_epd_fixtures.py -q'   # PASS
docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && python -m scripts.gen_epd_fixtures'
mkdir -p web/public/opening_book
docker exec 1ec2b8ce64c8 bash -lc 'cp /mnt/eloquence_bulk/databases/opening_book/band_*.json /workspaces/eloquent-encoding/web/public/opening_book/'
ls web/public/opening_book/   # band_1000.json ... band_1900.json
```

- [ ] **Step 5: Commit**

```bash
git add scripts/gen_epd_fixtures.py tests/dataset_generation/test_gen_epd_fixtures.py web/src/inference/__fixtures__/epd_cases.json web/public/opening_book/
git commit -m "feat: ship opening-book band JSONs + epd parity fixtures"
```

---

### Task 2: `epdKey` + `OpeningBook` (TS) — the parity gate

**Files:**
- Create: `web/src/inference/openingBook.ts`
- Test: `web/src/inference/openingBook.test.ts`

**Interfaces:**
- Consumes: `epd_cases.json` (Task 1).
- Produces:
  - `epdKey(board: Chess): string` — `"<placement> <turn> <castling> <ep>"`, ep = legal-ep target or `-`.
  - `eloToBand(elo: number): number`.
  - `class OpeningBook { totalGames: number; positions: Record<string, {n:number; moves:Record<string,number>}>; lookup(board: Chess, threshold: number, rand: () => number): {from:string; to:string; promotion?:string} | null }`.

- [ ] **Step 1: Write the failing test**

`web/src/inference/openingBook.test.ts`:

```ts
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding/web && npx vitest run src/inference/openingBook.test.ts'`
Expected: FAIL — cannot find module `./openingBook`.

- [ ] **Step 3: Implement**

`web/src/inference/openingBook.ts`:

```ts
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
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding/web && npx vitest run src/inference/openingBook.test.ts'`
Expected: PASS. (If the parity test fails on the legal-ep case, the bug is in `epdKey`'s ep handling — confirm the ep move's `.to` is used; do NOT change the fixture.)

- [ ] **Step 5: Commit**

```bash
git add web/src/inference/openingBook.ts web/src/inference/openingBook.test.ts
git commit -m "feat: TS epdKey (python-chess parity) + OpeningBook.lookup"
```

---

### Task 3: `OpeningBookSet` (lazy per-band fetch + cache)

**Files:**
- Modify: `web/src/inference/openingBook.ts`
- Test: `web/src/inference/openingBookSet.test.ts`

**Interfaces:**
- Consumes: `OpeningBook`, `eloToBand` (Task 2).
- Produces: `class OpeningBookSet { constructor(baseUrl: string, fetchFn?: typeof fetch); forElo(elo: number): Promise<OpeningBook | null> }` — fetches `${baseUrl}opening_book/band_${eloToBand(elo)}.json` once per band, caches the promise, returns `null` if the response is not ok / errors.

- [ ] **Step 1: Write the failing test**

`web/src/inference/openingBookSet.test.ts`:

```ts
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding/web && npx vitest run src/inference/openingBookSet.test.ts'`
Expected: FAIL — `OpeningBookSet` not exported.

- [ ] **Step 3: Implement**

Append to `web/src/inference/openingBook.ts`:

```ts
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding/web && npx vitest run src/inference/openingBookSet.test.ts'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/src/inference/openingBook.ts web/src/inference/openingBookSet.test.ts
git commit -m "feat: OpeningBookSet (lazy per-band fetch + cache)"
```

---

### Task 4: `bookOrModelMove` + wire into the bot

**Files:**
- Create: `web/src/inference/bookMove.ts`
- Test: `web/src/inference/bookMove.test.ts`
- Modify: `web/src/useEngine.ts`, `web/src/components/BoardPanel.tsx`

**Interfaces:**
- Consumes: `OpeningBookSet` (Task 3), `Engine` (existing).
- Produces: `bookOrModelMove(books: OpeningBookSet | null, engine: Engine, board: Chess, elo: number, opts: { temperature: number; greedy?: boolean; rand?: () => number }, threshold = 0.01): Promise<{from:string; to:string; promotion?:string}>` — tries the book first (`books.forElo(elo).lookup`), returns it on a hit; otherwise returns `engine.chooseMove(board, elo, opts)`.

- [ ] **Step 1: Write the failing test**

`web/src/inference/bookMove.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { Chess } from "chess.js";
import { OpeningBook, OpeningBookSet, epdKey } from "./openingBook";
import { bookOrModelMove } from "./bookMove";

const start = epdKey(new Chess());
function setReturning(mv: any) {
  const set = new OpeningBookSet("/b/");
  // @ts-expect-error inject a resolved book for the test
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding/web && npx vitest run src/inference/bookMove.test.ts'`
Expected: FAIL — cannot find module `./bookMove`.

- [ ] **Step 3: Implement the helper + wire it in**

`web/src/inference/bookMove.ts`:

```ts
import type { Chess } from "chess.js";
import type { Engine } from "./engine";
import type { OpeningBookSet, BookMove } from "./openingBook";

export const BOOK_THRESHOLD = 0.01;

export async function bookOrModelMove(
  books: OpeningBookSet | null, engine: Engine, board: Chess, elo: number,
  opts: { temperature: number; greedy?: boolean; rand?: () => number }, threshold = BOOK_THRESHOLD,
): Promise<BookMove> {
  if (books) {
    const bk = await books.forElo(elo);
    const mv = bk?.lookup(board, threshold, Math.random);
    if (mv) return mv;
  }
  return engine.chooseMove(board, elo, opts);
}
```

`web/src/useEngine.ts` — construct the book set alongside the engine and return it:

```tsx
import { OpeningBookSet } from "./inference/openingBook";
// inside useEngine, after computing `base = import.meta.env.BASE_URL`:
const [books] = useState(() => new OpeningBookSet(import.meta.env.BASE_URL));
// return { engine, error, books };
```

`web/src/components/BoardPanel.tsx` — `botMove` uses the helper (replace the direct `engine.chooseMove`):

```tsx
import { bookOrModelMove } from "../inference/bookMove";
// BoardPanel now also receives `books: OpeningBookSet | null` as a prop from App (threaded from useEngine).
// in botMove, replace `const mv = await engine.chooseMove(g, elo, {...})` with:
const mv = await bookOrModelMove(books, engine, g, elo, { temperature, greedy: false });
```

Thread `books` from `App` (which calls `useEngine`) into `<BoardPanel books={books} ... />`.

- [ ] **Step 4: Run tests + build**

```bash
docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding/web && npx vitest run src/inference/bookMove.test.ts && npm run build'
```
Expected: helper tests PASS; production build succeeds (typechecks the `books` prop threading).

- [ ] **Step 5: Commit**

```bash
git add web/src/inference/bookMove.ts web/src/inference/bookMove.test.ts web/src/useEngine.ts web/src/components/BoardPanel.tsx
git commit -m "feat: bot consults opening book before the model"
```

---

## Self-Review

**Spec coverage:** keep epd keys / no re-key (constraints + Task 1 ships existing JSONs); ep-normalized TS key (Task 2 `epdKey`); parity fixture as the gating test (Task 1 fixture + Task 2 parity test); `OpeningBook.lookup` ∝ counts, legal-filtered, threshold gate (Task 2); lazy per-band fetch + null on missing (Task 3 `OpeningBookSet`); book-before-model, off-cleanly, default 1% (Task 4 `bookOrModelMove` + wiring); ship all 10 bands committed (Task 1). Panel unchanged (Task 4 only touches `botMove`). All covered.

**Placeholder scan:** every code step has complete code; commands have expected output. No TBD/TODO.

**Type consistency:** `epdKey(board)`/`eloToBand(elo)`/`OpeningBook(totalGames, positions)` (Task 2) reused in Tasks 3–4; `BookMove = {from,to,promotion?}` returned by `lookup` (Task 2) and `bookOrModelMove` (Task 4); `OpeningBookSet(baseUrl, fetchFn?)` + `forElo` (Task 3) consumed by Task 4; band JSON keys `total_games`/`positions` (constraint) read in `OpeningBookSet` (Task 3) and produced by the Python `save_all` (matches the build). UCI move encoding (`from+to+promotion`) consistent between the book contents and the `lookup` legal-move matching.
