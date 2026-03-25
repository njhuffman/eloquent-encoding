#!/usr/bin/env python3
"""
Stream a .pgn.zst file (decompress in memory, no temp file), count games by
100-point WhiteElo buckets from header lines only (no python-chess).

Only games whose [TimeControl "..."] matches the requested value (default
Lichess 10-minute: 600+0) are counted.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import zstandard
from tqdm import tqdm

# PGN tags (whole-line style, as in Lichess exports)
_EVENT_START = "[Event"
_TIME_CONTROL_RE = re.compile(r'\[TimeControl\s+"([^"]*)"\]', re.IGNORECASE)
_WHITE_ELO_RE = re.compile(r'\[WhiteElo\s+"(\d+)"\]', re.IGNORECASE)

# Rough compressed bytes per game for tqdm total hint (Lichess standard monthly dumps)
_BYTES_PER_GAME_ESTIMATE = 350

# 10 minutes, no increment (Lichess standard rated dumps)
DEFAULT_TIME_CONTROL = "600+0"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Histogram WhiteElo from a streamed .pgn.zst (100-point buckets)."
    )
    parser.add_argument(
        "pgn_zst",
        type=Path,
        help="Path to .pgn.zst (e.g. Lichess monthly standard dump)",
    )
    parser.add_argument(
        "--time-control",
        default=DEFAULT_TIME_CONTROL,
        metavar="TC",
        help=f'Only count games with this [TimeControl] value (default: "{DEFAULT_TIME_CONTROL}")',
    )
    args = parser.parse_args()
    time_control_filter = args.time_control
    zst_path = args.pgn_zst.resolve()
    if not zst_path.is_file():
        print(f"error: not a file: {zst_path}", file=sys.stderr)
        return 1
    if zst_path.suffix.lower() != ".zst":
        print("warning: expected a .zst file", file=sys.stderr)

    counts: dict[int, int] = defaultdict(int)
    matched_games = 0
    processed_games = 0
    cur_tc: str | None = None
    cur_white_elo: int | None = None
    seen_first_event = False

    try:
        file_size = os.path.getsize(zst_path)
        total_estimate = max(1, int(file_size / _BYTES_PER_GAME_ESTIMATE))
    except OSError:
        total_estimate = None

    out_path = zst_path.parent / f"{zst_path.stem}_elo_counts.json"

    with open(zst_path, "rb") as raw:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(raw) as reader:
            text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            with tqdm(
                total=total_estimate,
                unit=" games",
                unit_scale=True,
                desc=zst_path.name,
                file=sys.stderr,
                miniters=1,
                mininterval=0.1,
                dynamic_ncols=True,
            ) as pbar:

                def finalize_game() -> None:
                    nonlocal matched_games, processed_games, cur_tc, cur_white_elo
                    processed_games += 1
                    pbar.update(1)
                    if cur_tc == time_control_filter and cur_white_elo is not None:
                        b = (cur_white_elo // 100) * 100
                        counts[b] += 1
                        matched_games += 1
                    cur_tc = None
                    cur_white_elo = None
                    pbar.set_postfix(matched=matched_games, refresh=False)

                for line in text:
                    stripped = line.strip()
                    if stripped.startswith(_EVENT_START):
                        if seen_first_event:
                            finalize_game()
                        else:
                            seen_first_event = True
                        cur_tc = None
                        cur_white_elo = None
                        continue
                    if stripped.startswith("[TimeControl"):
                        m = _TIME_CONTROL_RE.match(stripped)
                        if m:
                            cur_tc = m.group(1)
                        continue
                    if "WhiteElo" not in stripped:
                        continue
                    m = _WHITE_ELO_RE.search(stripped)
                    if m:
                        cur_white_elo = int(m.group(1))

                if seen_first_event:
                    finalize_game()

    # JSON object keys are strings; bucket key = floor(elo/100)*100 (e.g. 1500 -> "1500" for 1500–1599)
    ordered = {str(k): counts[k] for k in sorted(counts)}
    payload = {
        "time_control": time_control_filter,
        "games_processed": processed_games,
        "games_matched": matched_games,
        "bucket_size": 100,
        "counts": ordered,
    }

    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
