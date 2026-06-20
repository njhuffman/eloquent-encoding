#!/usr/bin/env python3
"""
Build rfp HDF5 from a YAML recipe (same recipes as gfp / jepa3).

  python -m rfp.scripts.pgn_to_h5 build --recipe dataset_generation/training_1M.yaml \\
      --data-dir DATA --output-dir databases/moves \\
      --encoder-checkpoint jepa3_checkpoints/.../stage.pt \\
      --history-len 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from dataset_generation.recipe import Recipe

from rfp.build_stream import build_rfp_from_recipe
from rfp.h5_io import rfp_h5_row_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build rfp (Residual From Predictor) HDF5 from a YAML recipe."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Stream PGN.zst sources and write rfp HDF5")
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
    b.add_argument(
        "--encoder-checkpoint",
        type=Path,
        required=True,
        help="jepa3 checkpoint containing encoder_online.*",
    )
    b.add_argument(
        "--history-len",
        type=int,
        required=True,
        help="Number of delta-z positions (N); requires N+1 board encodes per sample.",
    )
    b.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda | cpu (default: auto)",
    )
    b.add_argument(
        "--encoder-strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Strict load encoder weights (default: true)",
    )

    args = parser.parse_args()
    if args.cmd == "build":
        recipe = Recipe.load(args.recipe)
        dev = None if args.device is None else torch.device(args.device)
        try:
            out = build_rfp_from_recipe(
                recipe,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                encoder_checkpoint=args.encoder_checkpoint,
                history_len=int(args.history_len),
                device=dev,
                encoder_strict=bool(args.encoder_strict),
            )
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        n = rfp_h5_row_count(out)
        target = recipe.target_sample_rows()
        print(out)
        print(f"rows={n}  recipe_target_rows={target}", file=sys.stderr)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
