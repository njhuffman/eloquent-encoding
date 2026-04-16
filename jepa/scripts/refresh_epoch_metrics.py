#!/usr/bin/env python3
"""
Append one JSONL row per stage by re-evaluating saved checkpoints (train+val forward only).

  python -m jepa.scripts.refresh_epoch_metrics --model example
  python -m jepa.scripts.refresh_epoch_metrics --all-models
  python -m jepa.scripts.refresh_epoch_metrics --model foo --stages 1 2
  python -m jepa.scripts.refresh_epoch_metrics --model foo --dry-run
  python -m jepa.scripts.refresh_epoch_metrics --model foo --rematerialize   # rebuild cached H5 if needed

Requires materialized train/val HDF5 under cache_dir (from a prior train run, or rematerialize).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from jepa.epoch_metrics_refresh import refresh_epoch_metrics_for_model, resolve_refresh_device
from jepa.model_spec import MODEL_CONFIGS_DIR, load_model_spec, spec_path_for_model


def _iter_spec_paths() -> list[Path]:
    paths: list[Path] = []
    if not MODEL_CONFIGS_DIR.is_dir():
        return paths
    for ext in (".yaml", ".yml"):
        paths.extend(sorted(MODEL_CONFIGS_DIR.glob(f"*{ext}")))
    return paths


def main() -> int:
    p = argparse.ArgumentParser(
        description="Append epoch metrics JSONL rows by evaluating checkpoints (no training)"
    )
    p.add_argument("--model", type=str, default=None, help="Model name (jepa/model_configs/{name}.yaml)")
    p.add_argument("--all-models", action="store_true", help="Run for every YAML spec in model_configs")
    p.add_argument(
        "--stages",
        type=int,
        nargs="*",
        default=None,
        help="Training stages 1..N (default: all stages that have checkpoints)",
    )
    p.add_argument("--dry-run", action="store_true", help="List stages that would run")
    p.add_argument(
        "--rematerialize",
        action="store_true",
        help="Rebuild materialized HDF5 cache for each stage (slow; use if cache missing)",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device for forward pass (default: auto)",
    )
    p.add_argument("--quiet", action="store_true", help="Less stderr output")
    args = p.parse_args()

    if args.all_models and args.model is not None:
        print("Error: do not combine --all-models with --model", file=sys.stderr)
        return 1
    if not args.all_models and args.model is None:
        print("Error: specify --model NAME or --all-models", file=sys.stderr)
        return 1

    device = resolve_refresh_device(args.device)
    quiet = bool(args.quiet)
    stages = list(args.stages) if args.stages is not None else None

    if args.model is not None:
        try:
            spec_path = spec_path_for_model(args.model)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        spec = load_model_spec(spec_path)
        if spec["name"] != args.model:
            print(f"Error: spec name {spec['name']!r} != --model {args.model!r}", file=sys.stderr)
            return 1
        done = refresh_epoch_metrics_for_model(
            spec,
            device=device,
            stages=stages,
            rematerialize=args.rematerialize,
            quiet=quiet,
            dry_run=args.dry_run,
        )
        if not quiet:
            print(
                ("Would refresh stages " if args.dry_run else "Refreshed stages ")
                + str(done)
                + f" for {spec['name']}",
                file=sys.stderr,
            )
        return 0

    n_ok = 0
    for sp in _iter_spec_paths():
        spec = load_model_spec(sp)
        name = spec["name"]
        try:
            done = refresh_epoch_metrics_for_model(
                spec,
                device=device,
                stages=stages,
                rematerialize=args.rematerialize,
                quiet=quiet,
                dry_run=args.dry_run,
            )
            if not quiet and args.dry_run:
                print(f"[dry-run] {name}: stages {done}", file=sys.stderr)
            n_ok += 1
        except Exception as e:
            print(f"Error for {name}: {e}", file=sys.stderr)
    if not args.dry_run and not quiet:
        print(f"Processed {n_ok} model(s).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
