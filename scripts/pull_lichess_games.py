#!/usr/bin/env python3
"""
Download Lichess standard monthly PGN dumps, filter by date range, Elo (both players),
and TimeControl (600+0), and write matching games to a single PGN file.
"""

import argparse
import io
import os
import re
import sys
from pathlib import Path

import requests
import zstandard
from tqdm import tqdm

LICHESS_BASE = "https://database.lichess.org/standard"
FILENAME_TEMPLATE = "lichess_db_standard_rated_{year:04d}-{month:02d}.pgn.zst"
TIME_CONTROL_FILTER = "600+0"


def parse_date(s: str) -> tuple[int, int]:
    """Parse 'YYYY-MM' into (year, month)."""
    parts = s.strip().split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid date format: {s!r}. Use YYYY-MM.")
    year, month = int(parts[0]), int(parts[1])
    if not (1 <= month <= 12):
        raise ValueError(f"Month must be 1-12, got {month}")
    return year, month


def month_range(from_date: str, to_date: str) -> list[tuple[int, int]]:
    """Yield (year, month) for every month in [from_date, to_date] inclusive (YYYY-MM)."""
    y1, m1 = parse_date(from_date)
    y2, m2 = parse_date(to_date)
    if (y1, m1) > (y2, m2):
        raise ValueError(f"--from {from_date} must be <= --to {to_date}")
    out = []
    y, m = y1, m1
    while (y, m) <= (y2, m2):
        out.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def download_zst(url: str, path: Path, skip_existing: bool) -> bool:
    """Download url to path. Return True if file was downloaded, False if skipped."""
    if skip_existing and path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=2**20):
            if chunk:
                f.write(chunk)
    return True


# Header tag pattern: [TagName "value"] -> capture value (no quotes, value can contain + etc.)
_HEADER_VALUE_RE = re.compile(r'\[(\w+)\s+"([^"]*)"\]')


def _game_passes_filter(
    game_lines: list[str],
    time_control: str,
    elo_min: int | None,
    elo_max: int | None,
) -> bool:
    """Scan game lines for [TimeControl "..."] and [WhiteElo/BlackElo "..."]; return True if filters pass."""
    time_control_ok = False
    white_elo: int | None = None
    black_elo: int | None = None
    for line in game_lines:
        stripped = line.strip()
        if not stripped.startswith("["):
            continue
        m = _HEADER_VALUE_RE.match(stripped)
        if not m:
            continue
        tag, value = m.group(1), m.group(2)
        if tag == "TimeControl":
            time_control_ok = value == time_control
        elif tag == "WhiteElo":
            try:
                white_elo = int(value)
            except ValueError:
                pass
        elif tag == "BlackElo":
            try:
                black_elo = int(value)
            except ValueError:
                pass
    if not time_control_ok:
        return False
    if elo_min is not None or elo_max is not None:
        if white_elo is None or black_elo is None:
            return False
        if elo_min is not None and (white_elo < elo_min or black_elo < elo_min):
            return False
        if elo_max is not None and (white_elo > elo_max or black_elo > elo_max):
            return False
    return True


def iter_games_fast(stream: io.TextIOWrapper):
    """
    Yield one game at a time as a list of lines (including newlines).
    Detects game end when we see a blank line after having seen a move line (starts with "1.").
    """
    buffer: list[str] = []
    seen_move_line = False
    for line in stream:
        if not line.strip():  # blank line
            if seen_move_line:
                buffer.append(line)
                yield buffer
                buffer = []
                seen_move_line = False
            else:
                buffer.append(line)
        else:
            buffer.append(line)
            if line.strip().startswith("1."):
                seen_move_line = True
    if buffer and seen_move_line:
        yield buffer


# Rough bytes per game in Lichess standard .zst (for progress bar total estimate)
BYTES_PER_GAME_ESTIMATE = 350


def process_month(
    zst_path: Path,
    out_handle,
    elo_min: int | None,
    elo_max: int | None,
    time_control: str,
    total_processed_so_far: int,
    total_kept_so_far: int,
) -> tuple[int, int]:
    """Process one .zst file: stream decompress, parse PGN, filter, write. Returns (processed, kept)."""
    processed = 0
    kept = 0
    try:
        file_size = os.path.getsize(zst_path)
        total_estimate = max(1, int(file_size / BYTES_PER_GAME_ESTIMATE))
    except OSError:
        total_estimate = None
    with open(zst_path, "rb") as f:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
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
                for game_lines in iter_games_fast(text):
                    processed += 1
                    if _game_passes_filter(game_lines, time_control, elo_min, elo_max):
                        kept += 1
                        out_handle.writelines(game_lines)
                        out_handle.write("\n")
                        out_handle.flush()
                    pbar.update(1)
                    pbar.set_postfix(kept=total_kept_so_far + kept, refresh=False)
    return processed, kept


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pull Lichess standard games by date range, filter by Elo and TimeControl 600+0, write to one PGN file."
    )
    parser.add_argument("--from", dest="from_date", required=True, metavar="YYYY-MM", help="Start month (inclusive)")
    parser.add_argument("--to", dest="to_date", required=True, metavar="YYYY-MM", help="End month (inclusive)")
    parser.add_argument("--elo-min", type=int, default=None, help="Min Elo for both White and Black")
    parser.add_argument("--elo-max", type=int, default=None, help="Max Elo for both White and Black")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output PGN path (default: databases/lichess_<from>_<to>_<elo_min>_<elo_max>.pgn)",
    )
    parser.add_argument(
        "--databases-dir",
        type=Path,
        default=Path("databases"),
        help="Directory for downloaded .zst files (default: databases)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip re-downloading .zst files that already exist",
    )
    args = parser.parse_args()

    try:
        months = month_range(args.from_date, args.to_date)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.output is None:
        elo_suffix = f"{args.elo_min or 0}_{args.elo_max or 9999}"
        args.output = Path("databases") / f"lichess_{args.from_date}_{args.to_date}_{elo_suffix}.pgn"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_processed = 0
    total_kept = 0

    with open(args.output, "w", encoding="utf-8") as out_file:
        for year, month in months:
            filename = FILENAME_TEMPLATE.format(year=year, month=month)
            url = f"{LICHESS_BASE}/{filename}"
            zst_path = args.databases_dir / filename
            print(f"Month {year:04d}-{month:02d}: downloading ...", file=sys.stderr)
            try:
                downloaded = download_zst(url, zst_path, args.skip_existing)
                if downloaded:
                    print(f"  Downloaded to {zst_path}", file=sys.stderr)
                else:
                    print(f"  Using existing {zst_path}", file=sys.stderr)
            except requests.RequestException as e:
                print(f"  Download failed: {e}", file=sys.stderr)
                continue
            p, k = process_month(
                zst_path,
                out_file,
                args.elo_min,
                args.elo_max,
                TIME_CONTROL_FILTER,
                total_processed_so_far=total_processed,
                total_kept_so_far=total_kept,
            )
            total_processed += p
            total_kept += k
    print(file=sys.stderr)
    print(f"Total processed: {total_processed}, Total kept: {total_kept}", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
