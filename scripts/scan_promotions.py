#!/usr/bin/env python3
"""Fast string-only scan of a lichess .pgn.zst for promotions, by elo band. No python-chess.

Promotions in PGN movetext are `=Q`/`=R`/`=B`/`=N` tokens; elo from [WhiteElo]/[BlackElo].
Each promotion is attributed to the promoting player (ply parity) and bucketed by that player's
100-Elo band. Ply counts use the per-move clock comment count ({-count) for speed; only games
containing '=' get tokenized for promotion attribution.
"""
from __future__ import annotations
import argparse, io, re, time
from collections import defaultdict
import zstandard

PROMO = re.compile(r"=([QRBN])")
COMMENT = re.compile(r"\{[^}]*\}")
MOVENUM = re.compile(r"\d+\.+")
RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}


def bandof(elo: int) -> int:
    return (elo // 100) * 100


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="/mnt/eloquence_bulk/databases/lichess_db_standard_rated_2025-01_tc_600_0.pgn.zst")
    ap.add_argument("--max-games", type=int, default=0, help="0 = whole file")
    ap.add_argument("--progress-every", type=int, default=2000000)
    a = ap.parse_args()

    plies = defaultdict(int); promos = defaultdict(int); under = defaultdict(int)
    piece = defaultdict(lambda: defaultdict(int))
    games = 0; we = be = None; mt_parts = []
    t0 = time.time()

    def flush():
        nonlocal we, be, games, mt_parts
        if not mt_parts:
            we = be = None; return
        games += 1
        raw = " ".join(mt_parts)
        nclk = raw.count("{")  # one clock comment per ply
        if we is not None:
            plies[bandof(we)] += (nclk + 1) // 2
        if be is not None:
            plies[bandof(be)] += nclk // 2
        if "=" in raw:
            mt = MOVENUM.sub(" ", COMMENT.sub(" ", raw))
            toks = [t for t in mt.split() if t not in RESULTS and t[0] != "$"]
            for i, t in enumerate(toks):
                m = PROMO.search(t)
                if not m:
                    continue
                elo = we if (i % 2 == 0) else be
                if elo is None:
                    continue
                b = bandof(elo); promos[b] += 1
                if m.group(1) != "Q":
                    under[b] += 1; piece[b][m.group(1)] += 1
        mt_parts = []; we = be = None

    with open(a.file, "rb") as fh:
        reader = zstandard.ZstdDecompressor().stream_reader(fh)
        for line in io.TextIOWrapper(reader, encoding="utf-8", errors="replace"):
            if line[0:1] == "[":
                if line.startswith("[Event ") and mt_parts:
                    flush()
                    if a.max_games and games >= a.max_games:
                        break
                    if games % a.progress_every == 0:
                        print(f"  ...{games:,} games ({games/(time.time()-t0):.0f}/s)", flush=True)
                elif line.startswith("[WhiteElo "):
                    try: we = int(line.split('"')[1])
                    except Exception: we = None
                elif line.startswith("[BlackElo "):
                    try: be = int(line.split('"')[1])
                    except Exception: be = None
            elif line[0:1] not in ("\n", "\r", "", " "):
                mt_parts.append(line.strip())
        flush()

    tot_pl = sum(plies.values()); tot_pr = sum(promos.values()); tot_un = sum(under.values())
    pc = defaultdict(int)
    for b in piece:
        for k, v in piece[b].items():
            pc[k] += v
    print(f"\nscanned {games:,} games, {tot_pl:,} plies in {time.time()-t0:.0f}s ({a.file.split('/')[-1]})")
    print(f"promotions:      {tot_pr:,}   ({1000*tot_pr/max(tot_pl,1):.2f} per 1000 plies)")
    print(f"underpromotions: {tot_un:,}   = {100*tot_un/max(tot_pr,1):.2f}% of promotions, "
          f"{1e5*tot_un/max(tot_pl,1):.2f} per 100k plies")
    print(f"under pieces:    " + ", ".join(f"={k} {pc[k]:,}" for k in ("N", "R", "B")))
    print(f"\n{'band':>6} {'plies':>13} {'promos':>9} {'under':>7} {'und%ofpromo':>12} {'und/100kply':>12}")
    for b in sorted(plies):
        if plies[b] <= 0:
            continue
        pr = promos[b]; un = under[b]
        print(f"{b:>6} {plies[b]:>13,} {pr:>9,} {un:>7,} "
              f"{(100*un/pr if pr else 0):>11.2f}% {1e5*un/plies[b]:>12.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
