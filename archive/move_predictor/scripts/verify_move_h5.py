#!/usr/bin/env python3
"""
Quick read-only scan of move-predictor train HDF5: stripe-read every dataset row range.
Exits 0 if all reads succeed, 1 on first OSError (likely bad storage / corrupt chunk at that offset).

Usage:
  python -m move_predictor.scripts.verify_move_h5 databases/moves/1M_mayfly/train.h5
  python -m move_predictor.scripts.verify_move_h5 train.h5 --stripe-rows 4096
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py


def _scan_dataset(name: str, ds, stripe: int) -> None:
    shape = ds.shape
    n = shape[0]
    print(f"  {name} shape={shape} ...", flush=True)
    for start in range(0, n, stripe):
        end = min(start + stripe, n)
        try:
            _ = ds[start:end]
        except OSError as e:
            print(f"\nFAILED {name} rows [{start}, {end}): {e}", file=sys.stderr)
            raise SystemExit(1)


def main() -> int:
    p = argparse.ArgumentParser(description="Stripe-read move HDF5 to detect EIO/corruption")
    p.add_argument("h5", type=Path, help="Path to train.h5 (or any move HDF5)")
    p.add_argument("--stripe-rows", type=int, default=8192, help="Rows per read stripe")
    args = p.parse_args()

    if not args.h5.is_file():
        print(f"Error: not found: {args.h5}", file=sys.stderr)
        return 1

    keys = (
        "cur_emb",
        "hist_white_emb",
        "hist_black_emb",
        "hist_white_len",
        "hist_black_len",
        "side_to_move",
        "from_sq",
        "to_sq",
        "label",
        "fen",
    )
    with h5py.File(args.h5, "r") as f:
        for k in keys:
            if k not in f:
                print(f"Warning: missing dataset {k!r}", file=sys.stderr)
                continue
            if k == "fen":
                ds = f[k]
                n = ds.shape[0]
                print(f"  {k} n={n} (vlen) ...", flush=True)
                for start in range(0, n, args.stripe_rows):
                    end = min(start + args.stripe_rows, n)
                    try:
                        _ = ds[start:end]
                    except OSError as e:
                        print(f"\nFAILED {k} rows [{start}, {end}): {e}", file=sys.stderr)
                        return 1
            else:
                _scan_dataset(k, f[k], args.stripe_rows)

    print("OK: all stripe reads succeeded (file is readable end-to-end).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
