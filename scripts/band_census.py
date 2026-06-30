"""Quick per-band game census for mining recipes.

Header-only scan (no python-chess parsing) at ~200k games/s, so a few months take ~1 min.
Buckets games by the WhiteElo header (matching recipe `bucket_by: white`) into 100-pt bands
and reports per-source + combined counts, so you can confirm each band can satisfy a target
`take_games` BEFORE committing to a full mine. Counts are a slight upper bound on usable games
(they don't apply the >= samples_per_game candidate-position filter, which drops ~2% of games).

Usage:
  python scripts/band_census.py --data-dir /mnt/eloquence_bulk/databases \
      --sources lichess_db_standard_rated_2025-01_tc_600_0.pgn.zst ... \
      [--band-min 1000 --band-max 2200 --band-width 100] [--target 444000]
"""
from __future__ import annotations

import argparse
import io
import re
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import zstandard

_WHITE_ELO = re.compile(r'^\[WhiteElo "([^"]*)"\]')


def _census_one(args: tuple[str, int, int, int]) -> tuple[str, Counter, int, int]:
    """Scan one .zst, return (source, band_lo -> count, total_games, unrated_or_oob)."""
    path, band_min, band_max, band_width = args
    counts: Counter = Counter()
    total = 0
    other = 0  # rated value present but outside [band_min, band_max) or unparseable/unrated
    dctx = zstandard.ZstdDecompressor()
    reader = dctx.stream_reader(open(path, "rb"))
    text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    for line in text:
        if not line.startswith("[WhiteElo "):
            continue
        m = _WHITE_ELO.match(line)
        if m is None:
            continue
        total += 1
        raw = m.group(1)
        try:
            elo = int(raw)
        except ValueError:
            other += 1
            continue
        if band_min <= elo < band_max:
            lo = band_min + ((elo - band_min) // band_width) * band_width
            counts[lo] += 1
        else:
            other += 1
    return Path(path).name, counts, total, other


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-band game census (header-only scan).")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--band-min", type=int, default=1000)
    ap.add_argument("--band-max", type=int, default=2200)
    ap.add_argument("--band-width", type=int, default=100)
    ap.add_argument("--target", type=int, default=None,
                    help="Optional per-(band,source) take_games target to flag bands that fall short.")
    a = ap.parse_args()

    jobs = [(str(a.data_dir / s), a.band_min, a.band_max, a.band_width) for s in a.sources]
    with Pool(processes=len(jobs)) as pool:
        results = pool.map(_census_one, jobs)

    bands = list(range(a.band_min, a.band_max, a.band_width))
    per_source = {name: counts for name, counts, _, _ in results}
    names = [name for name, *_ in results]

    width = max(12, *(len(n) for n in names))
    hdr = f'{"band":>11} | ' + " | ".join(f"{n[-7:]:>9}" for n in names) + " | " + f'{"COMBINED":>11}'
    print(hdr)
    print("-" * len(hdr))
    combined_totals = {}
    for lo in bands:
        row = f"{lo}-{lo + a.band_width - 1:>4} | "
        combo = 0
        cells = []
        for n in names:
            c = per_source[n].get(lo, 0)
            combo += c
            cells.append(f"{c:>9,}")
        combined_totals[lo] = combo
        flag = ""
        if a.target is not None:
            need = a.target * len(names)
            flag = "  OK" if combo >= need else f"  SHORT (need {need:,})"
        print(row + " | ".join(cells) + " | " + f"{combo:>11,}" + flag)

    print("-" * len(hdr))
    for name, counts, total, other in results:
        in_band = sum(counts.values())
        print(f"{name}: total={total:,}  in-band={in_band:,}  unrated/out-of-range={other:,}")
    grand = sum(combined_totals.values())
    print(f"\nCombined in-band games [{a.band_min},{a.band_max}): {grand:,}  "
          f"-> ~{grand * 8:,} samples at 8/game (upper bound; ~2% lost to short games)")
    if combined_totals:
        scarce_lo = min(combined_totals, key=lambda k: combined_totals[k])
        print(f"Scarcest band: {scarce_lo}-{scarce_lo + a.band_width - 1} "
              f"with {combined_totals[scarce_lo]:,} games "
              f"-> even-spread is capped at {combined_totals[scarce_lo] * 8 * len(bands):,} total samples "
              f"({len(bands)} bands).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
