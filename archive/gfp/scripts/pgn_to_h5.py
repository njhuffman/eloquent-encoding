#!/usr/bin/env python3
"""
Build gfp slim HDF5 from a YAML recipe (same recipes as dataset_generation / jepa3).

  python -m gfp.scripts.pgn_to_h5 build --recipe dataset_generation/training_1M.yaml \\
      --data-dir DATA --output-dir databases/moves
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dataset_generation.recipe import Recipe

from gfp.build_stream import build_gfp_from_recipe
from gfp.h5_io import gfp_h5_row_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build gfp (Global From Predictor) HDF5 from a YAML recipe."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Stream PGN.zst sources and write gfp HDF5")
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
            out = build_gfp_from_recipe(
                recipe,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
            )
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        n = gfp_h5_row_count(out)
        target = recipe.target_sample_rows()
        print(out)
        print(f"rows={n}  recipe_target_rows={target}", file=sys.stderr)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
