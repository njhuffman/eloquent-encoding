from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py

from dataset_generation.builder import build_from_recipe
from dataset_generation.recipe import Recipe


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build move-sample HDF5 from a YAML recipe (JSON also valid YAML)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Stream PGN.zst sources and write HDF5")
    b.add_argument(
        "--recipe",
        type=Path,
        required=True,
        help="Path to recipe file (.yaml / .yml recommended; supports # comments)",
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
            out = build_from_recipe(
                recipe,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
            )
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        target = recipe.target_sample_rows()
        with h5py.File(out, "r") as f:
            n = int(f["fen"].shape[0])
        print(out)
        print(f"rows={n}  recipe_target_rows={target}", file=sys.stderr)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
