#!/usr/bin/env python3
"""
Recompute jepa3 ``metrics/<name>_stage_benchmarks.json`` (joint from+to top-1 on packed HDF5).

  python -m jepa3.scripts.refresh_dashboard_metrics --model tiny_test
  python -m jepa3.scripts.refresh_dashboard_metrics --model tiny_test --stages 0 1
  python -m jepa3.scripts.refresh_dashboard_metrics --model tiny_test --sample-n 256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from jepa.h5_bootstrap import apply_hdf5_read_safety_env

apply_hdf5_read_safety_env()

from jepa3.dashboard_metrics import refresh_dashboard_metrics_for_model
from jepa3.model_spec import MODEL_CONFIGS_DIR, load_model_spec, spec_path_for_model


def _iter_spec_paths() -> list[Path]:
    paths: list[Path] = []
    if not MODEL_CONFIGS_DIR.is_dir():
        return paths
    for ext in (".yaml", ".yml"):
        paths.extend(sorted(MODEL_CONFIGS_DIR.glob(f"*{ext}")))
    return paths


def main() -> int:
    p = argparse.ArgumentParser(
        description="Refresh jepa3 stage benchmark JSON (joint from+to top1 on packed val/train sample)"
    )
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--all-models", action="store_true")
    p.add_argument("--stages", type=int, nargs="*", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Rows per forward batch for joint top1 eval (default 256)",
    )
    p.add_argument(
        "--sample-n",
        type=int,
        default=None,
        help="Override spec dashboard_metrics.move_benchmark_sample_n (e.g. 256 for a quick pass)",
    )
    args = p.parse_args()

    if args.all_models and args.model is not None:
        print("Error: do not combine --all-models with --model", file=sys.stderr)
        return 1
    if not args.all_models and args.model is None:
        print("Error: specify --model NAME or --all-models", file=sys.stderr)
        return 1

    quiet = not args.verbose

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
        stages = list(args.stages) if args.stages is not None else None
        refresh_dashboard_metrics_for_model(
            spec,
            stages=stages,
            quiet=quiet,
            dry_run=args.dry_run,
            eval_batch_size=max(1, int(args.batch_size)),
            sample_n_override=args.sample_n,
        )
        return 0

    n_ok = 0
    for sp in _iter_spec_paths():
        spec = load_model_spec(sp)
        stages = list(args.stages) if args.stages is not None else None
        try:
            refresh_dashboard_metrics_for_model(
                spec,
                stages=stages,
                quiet=quiet,
                dry_run=args.dry_run,
                eval_batch_size=max(1, int(args.batch_size)),
                sample_n_override=args.sample_n,
            )
            n_ok += 1
        except Exception as e:
            print(f"Error for {spec['name']}: {e}", file=sys.stderr)
    if not args.dry_run:
        print(f"Processed {n_ok} model(s).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
