"""
Single-process read-through check for JEPA materialized HDF5 caches.

Use after EIO / flaky-storage errors to see if the file is readable without
rematerializing. Example:

  python -m jepa.scripts.verify_materialized_h5 \\
    jepa_checkpoints/jepa_y_1M/cache/jepa_y_stage3_4413abe82b91_train.h5

``--quick`` only touches start/middle/end row windows (seconds); ``--full`` scans
every row in ``--chunk-rows`` steps (minutes for multi-GB files, but still only I/O).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from jepa.h5_bootstrap import apply_hdf5_read_safety_env

apply_hdf5_read_safety_env()

import h5py
import numpy as np


def _read_slice(f: h5py.File, start: int, end: int) -> None:
    sl = slice(start, end)
    _ = np.asarray(f["board_t"][sl], dtype=np.float32)
    _ = np.asarray(f["board_t_plus_1_pos"][sl], dtype=np.float32)
    _ = np.asarray(f["board_t_plus_1_negs"][sl], dtype=np.float32)
    _ = np.asarray(f["elo"][sl], dtype=np.float32)


def main() -> int:
    p = argparse.ArgumentParser(description="Verify JEPA materialized HDF5 is readable (single process).")
    p.add_argument("h5_path", type=Path, help="Path to materialized .h5")
    p.add_argument("--chunk-rows", type=int, default=1024, help="Row batch size for full scan")
    p.add_argument(
        "--mode",
        choices=("quick", "full"),
        default="full",
        help="quick: start/middle/end windows only; full: sequential scan",
    )
    args = p.parse_args()
    path = args.h5_path.expanduser().resolve()
    if not path.is_file():
        print(f"error: not a file: {path}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    with h5py.File(path, "r") as f:
        n = int(f["board_t"].shape[0])
        chunk = max(1, int(args.chunk_rows))
        print(f"{path}")
        print(f"rows={n} mode={args.mode} chunk_rows={chunk}")

        if args.mode == "quick":
            if n == 0:
                print("ok (empty)")
                return 0
            windows: list[tuple[int, int]] = []
            w = min(chunk, n)
            windows.append((0, w))
            if n > w:
                mid = max(0, n // 2 - w // 2)
                windows.append((mid, min(mid + w, n)))
            if n > w:
                windows.append((max(0, n - w), n))
            for a, b in windows:
                print(f"  read rows [{a}, {b})")
                _read_slice(f, a, b)
        else:
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                if start == 0 or end == n or (start // chunk) % 100 == 0:
                    print(f"  rows [{start}, {end})")
                _read_slice(f, start, end)

    dt = time.perf_counter() - t0
    print(f"ok ({dt:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
