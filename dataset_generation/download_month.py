#!/usr/bin/env python3
"""
Stream-download a Lichess standard rated monthly .pgn.zst, keep only games matching
a [TimeControl] value, and write a single compressed .pgn.zst (no full dump on disk).

Example:
  python -m dataset_generation.download_month 2013-01 --time-control 600+8 --output-dir databases
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import urllib.request
from pathlib import Path

import zstandard
from tqdm import tqdm

from dataset_generation.pgn_filter import game_text_matches_time_control
from dataset_generation.stream import iter_pgn_game_texts

LICHESS_STANDARD_RATED = "https://database.lichess.org/standard"
_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")


def time_control_filename_slug(time_control: str) -> str:
    """e.g. 600+8 -> 600_8; safe for filenames (no +, /, etc.)."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", time_control.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def default_output_basename(month: str, time_control: str) -> str:
    slug = time_control_filename_slug(time_control)
    return f"lichess_db_standard_rated_{month}_tc_{slug}.pgn.zst"


def lichess_month_url(month: str) -> str:
    return f"{LICHESS_STANDARD_RATED}/lichess_db_standard_rated_{month}.pgn.zst"


def download_filtered_month(
    month: str,
    time_control: str,
    output_path: Path,
    *,
    user_agent: str = "eloquence-dataset-download/1.0 (+https://github.com/)",
) -> tuple[int, int]:
    """
    Stream from Lichess, filter games by exact TimeControl header match, zstd-compress to output_path.
    Returns (games_matched, games_seen).
    """
    if not _MONTH_RE.match(month):
        raise ValueError(f"month must be YYYY-MM, got {month!r}")

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    url = lichess_month_url(month)
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})

    games_seen = 0
    games_kept = 0

    with urllib.request.urlopen(req, timeout=120) as resp:
        dctx = zstandard.ZstdDecompressor()
        cctx = zstandard.ZstdCompressor()

        with open(output_path, "wb") as fout:
            with cctx.stream_writer(fout) as zout:
                with dctx.stream_reader(resp) as reader:
                    text_in = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
                    pbar = tqdm(iter_pgn_game_texts(text_in), desc=f"{month} filter", unit=" games")
                    for game_text in pbar:
                        games_seen += 1
                        if game_text_matches_time_control(game_text, time_control):
                            zout.write(game_text.encode("utf-8"))
                            games_kept += 1
                        pbar.set_postfix(kept=games_kept, refresh=False)

    if games_kept == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"no games matched TimeControl {time_control!r} "
            f"(scanned {games_seen} games from {url})"
        )

    return games_kept, games_seen


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download a Lichess rated month, filter by TimeControl, write one .pgn.zst."
    )
    parser.add_argument(
        "month",
        help="Calendar month YYYY-MM (matches Lichess dump lichess_db_standard_rated_YYYY-MM)",
    )
    parser.add_argument(
        "--time-control",
        required=True,
        metavar="TC",
        help='Exact [TimeControl] value to keep, e.g. "600+0" or "600+8"',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the output .zst (created if missing)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        metavar="NAME",
        help="Override output filename (must end with .pgn.zst). Default encodes time control in the name.",
    )

    args = parser.parse_args()
    month = args.month.strip()
    tc = args.time_control.strip()

    if args.output_name is not None:
        name = args.output_name.strip()
        if not name.endswith(".pgn.zst"):
            print("error: --output-name must end with .pgn.zst", file=sys.stderr)
            return 1
        out = args.output_dir / name
    else:
        out = args.output_dir / default_output_basename(month, tc)

    try:
        kept, seen = download_filtered_month(month, tc, out)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(out)
    print(f"games_kept={kept} games_seen={seen}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
