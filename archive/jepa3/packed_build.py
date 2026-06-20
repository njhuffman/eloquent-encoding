#!/usr/bin/env python3
"""
Build jepa3 packed move HDF5 from a YAML recipe (same recipes as dataset_generation).

  python -m jepa3.packed_build build --recipe dataset_generation/training_1M.yaml \\
      --data-dir DATA --output-dir databases/moves

Profiling example (py-spy, scalene, etc.):

  python -m jepa3.packed_build build --recipe jepa3/j3_bench.yaml \\
      --data-dir /mnt/eloquence_bulk/databases/ --output-dir /mnt/eloquence_bulk/databases/

Writes ``{output_dir}/{recipe.name}.h5`` with attrs ``jepa3_packed_format=1``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dataset_generation.recipe import Recipe

from jepa3.packed_build_stream import build_packed_from_recipe
from jepa3.packed_h5 import packed_h5_row_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build jepa3 packed move-sample HDF5 from a YAML recipe."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Stream PGN.zst sources and write packed HDF5")
    b.add_argument(
        "--recipe",
        type=Path,
        required=True,
        help="Path to recipe file (.yaml / .yml)",
    )
    b.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing each source_plan `source` path (local .pgn.zst only)",
    )
    b.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the HDF5 file; recipe field `name` -> {output-dir}/{name}.h5",
    )

    args = parser.parse_args()
    if args.cmd == "build":
        recipe = Recipe.load(args.recipe)
        try:
            out = build_packed_from_recipe(
                recipe,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
            )
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        n = packed_h5_row_count(out)
        target = recipe.target_sample_rows()
        print(out)
        print(f"rows={n}  recipe_target_rows={target}", file=sys.stderr)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
