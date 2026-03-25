#!/usr/bin/env python3
"""
Load several *_elo_counts.json files (from pgn_zst_white_elo_histogram.py) and
print / optionally write the summed bucket counts and game totals.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_one(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"error reading {path}: {e}") from e
    for key in ("counts", "bucket_size", "games_processed", "games_matched"):
        if key not in data:
            raise SystemExit(f"error: {path} missing {key!r}")
    if not isinstance(data["counts"], dict):
        raise SystemExit(f"error: {path} counts must be an object")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate WhiteElo histogram JSON files into one summary."
    )
    parser.add_argument(
        "count_json",
        nargs="+",
        type=Path,
        help="Paths to *_elo_counts.json files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write aggregated JSON to this path (default: stdout only)",
    )
    args = parser.parse_args()

    paths = [p.resolve() for p in args.count_json]
    merged_counts: dict[str, int] = defaultdict(int)
    total_processed = 0
    total_matched = 0
    bucket_size: int | None = None
    time_control: str | None = None
    sources: list[str] = []

    for path in paths:
        if not path.is_file():
            print(f"error: not a file: {path}", file=sys.stderr)
            return 1
        data = load_one(path)
        bs = data["bucket_size"]
        tc = data.get("time_control")
        if bucket_size is None:
            bucket_size = bs
            time_control = tc
        elif bs != bucket_size:
            print(
                f"error: bucket_size mismatch {path}: {bs} vs {bucket_size}",
                file=sys.stderr,
            )
            return 1
        if tc != time_control:
            print(
                f"error: time_control mismatch {path}: {tc!r} vs {time_control!r}",
                file=sys.stderr,
            )
            return 1

        total_processed += int(data["games_processed"])
        total_matched += int(data["games_matched"])
        sources.append(str(path))
        for k, v in data["counts"].items():
            merged_counts[str(k)] += int(v)

    ordered = {k: merged_counts[k] for k in sorted(merged_counts, key=int)}
    payload = {
        "sources": sources,
        "time_control": time_control,
        "bucket_size": bucket_size,
        "games_processed": total_processed,
        "games_matched": total_matched,
        "counts": ordered,
    }

    text = json.dumps(payload, indent=2) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output.resolve()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
