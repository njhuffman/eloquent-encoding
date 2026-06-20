"""Write dashboard JSON artifacts (CPU profile + move-ranking on val/train samples) after training or via refresh CLI."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import torch

from jepa.checkpoint_paths import stage_checkpoint_path
from jepa.config import BOARD_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH
from jepa.load import load_jepa_from_checkpoint
from jepa.metrics_paths import model_profile_json_path, stage_benchmarks_json_path
from jepa.scripts.prediction_benchmark import _sample_indices, run_move_and_positive_metrics_for_checkpoint

_MOVE_H5_REQUIRED_KEYS = ("fen", "from_sq", "to_sq", "promotion", "elo_to_move")


def _assert_move_h5_schema(f: h5py.File, path: Path) -> None:
    for key in _MOVE_H5_REQUIRED_KEYS:
        if key not in f:
            raise ValueError(f"{path} missing dataset {key!r}")


def skip_dashboard_metrics() -> bool:
    v = os.environ.get("JEPA_SKIP_DASHBOARD_METRICS", "").strip().lower()
    return v in ("1", "true", "yes")


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(x) for x in obj]
    return obj


def resolve_benchmark_device(device_setting: str) -> torch.device:
    if device_setting == "cpu":
        return torch.device("cpu")
    if device_setting == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def profile_cpu_single_forward(ckpt_path: Path, *, warmup: int = 3, repeats: int = 20) -> dict[str, Any]:
    device = torch.device("cpu")
    model = load_jepa_from_checkpoint(ckpt_path, device=device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    board = torch.randn(1, BOARD_HEIGHT, BOARD_WIDTH, BOARD_CHANNELS, dtype=torch.float32)
    elo = torch.randn(1, dtype=torch.float32) * 1500.0
    with torch.no_grad():
        for _ in range(warmup):
            model.forward_online(board, elo)
        times: list[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model.forward_online(board, elo)
            times.append(time.perf_counter() - t0)
    mean_s = sum(times) / max(len(times), 1)
    return {
        "n_parameters": n_params,
        "cpu_single_forward_seconds": mean_s,
        "warmup": warmup,
        "repeats": repeats,
        "batch_size": 1,
    }


def write_profile_json(spec: dict[str, Any], stage0_ckpt: Path, *, quiet: bool = True) -> None:
    _ = quiet
    ckpt_dir = Path(spec["checkpoint_dir"])
    path = model_profile_json_path(ckpt_dir, spec["name"])
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = profile_cpu_single_forward(stage0_ckpt.resolve())
    rec["model"] = spec["name"]
    rec["checkpoint_path"] = str(stage0_ckpt.resolve())
    rec["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(_json_sanitize(rec), indent=2), encoding="utf-8")


def upsert_stage_benchmarks(
    path: Path,
    model_name: str,
    meta: dict[str, Any],
    stage_row: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {"version": 2, "model": model_name, "stages": []}
    for k, v in meta.items():
        if k != "stages":
            data[k] = v
    data["model"] = model_name
    data["version"] = 2
    stages = data.get("stages")
    if not isinstance(stages, list):
        stages = []
    sk = int(stage_row["stage"])
    stages = [x for x in stages if int(x.get("stage", -1)) != sk]
    stages.append(_json_sanitize(stage_row))
    stages.sort(key=lambda x: int(x["stage"]))
    data["stages"] = stages
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def compute_move_benchmark_row(
    spec: dict[str, Any],
    stage: int,
    ckpt_path: Path,
    *,
    quiet: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    dm = spec["dashboard_metrics"]
    val_h5 = Path(spec["val_move_dataset_h5"])
    if not val_h5.is_file():
        return None
    dev = resolve_benchmark_device(str(dm["device"]))
    use_amp = dev.type == "cuda"
    ckpt_path = ckpt_path.resolve()
    succ_chunk = max(1, int(dm["move_benchmark_succ_chunk"]))
    sample_n = int(dm["move_benchmark_sample_n"])
    with h5py.File(val_h5, "r") as f:
        _assert_move_h5_schema(f, val_h5)
        n_total = int(f["fen"].shape[0])
        indices = _sample_indices(n_total, sample_n, int(dm["move_benchmark_seed"]))
        stats, _by_elo, pos_stats = run_move_and_positive_metrics_for_checkpoint(
            ckpt_path,
            f,
            indices,
            dev,
            use_amp=use_amp,
            succ_chunk=succ_chunk,
            quiet=quiet,
        )
    row = _json_sanitize(asdict(stats))
    row["median_l2_pred_to_pos_ema"] = _json_sanitize(pos_stats.median_l2_pred_to_pos_ema)
    row["median_l2_pred_to_pos_online"] = _json_sanitize(pos_stats.median_l2_pred_to_pos_online)
    row["stage"] = int(stage)
    row["checkpoint_path"] = str(ckpt_path)

    train_h5 = Path(spec["train_move_dataset_h5"])
    if train_h5.is_file():
        with h5py.File(train_h5, "r") as tf:
            _assert_move_h5_schema(tf, train_h5)
            n_train = int(tf["fen"].shape[0])
            t_indices = _sample_indices(n_train, sample_n, int(dm["move_benchmark_train_seed"]))
            train_stats, _t_elo, train_pos = run_move_and_positive_metrics_for_checkpoint(
                ckpt_path,
                tf,
                t_indices,
                dev,
                use_amp=use_amp,
                succ_chunk=succ_chunk,
                quiet=quiet,
            )
        train_row = _json_sanitize(asdict(train_stats))
        train_row["median_l2_pred_to_pos_ema"] = _json_sanitize(train_pos.median_l2_pred_to_pos_ema)
        train_row["median_l2_pred_to_pos_online"] = _json_sanitize(train_pos.median_l2_pred_to_pos_online)
        row["train"] = train_row

    meta = {
        "val_move_dataset_h5": str(val_h5),
        "sample_n": sample_n,
        "seed": int(dm["move_benchmark_seed"]),
        "succ_chunk": succ_chunk,
        "device_used": str(dev),
        "pred_pos_median_l2_geometry": "euclidean_l2",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if train_h5.is_file():
        meta["train_move_dataset_h5"] = str(train_h5)
        meta["train_seed"] = int(dm["move_benchmark_train_seed"])
    return row, meta


def run_after_stage_zero(spec: dict[str, Any], stage0_ckpt: Path, *, quiet: bool = True) -> None:
    if skip_dashboard_metrics():
        return
    try:
        write_profile_json(spec, stage0_ckpt, quiet=quiet)
    except Exception as e:
        print(f"Warning: dashboard profile failed: {e}", file=sys.stderr)
    try:
        out = compute_move_benchmark_row(spec, 0, stage0_ckpt, quiet=quiet)
        if out is None:
            print("Warning: dashboard move benchmark skipped (val_move_dataset_h5 missing)", file=sys.stderr)
            return
        row, meta = out
        p = stage_benchmarks_json_path(Path(spec["checkpoint_dir"]), spec["name"])
        upsert_stage_benchmarks(p, spec["name"], meta, row)
    except Exception as e:
        print(f"Warning: dashboard stage 0 move benchmark failed: {e}", file=sys.stderr)


def run_after_training_stage(spec: dict[str, Any], stage: int, ckpt_path: Path, *, quiet: bool = True) -> None:
    if skip_dashboard_metrics():
        return
    ckpt_dir = Path(spec["checkpoint_dir"])
    p0 = stage_checkpoint_path(ckpt_dir, spec["name"], 0)
    if p0.is_file():
        try:
            write_profile_json(spec, p0, quiet=quiet)
        except Exception as e:
            print(f"Warning: dashboard profile refresh failed: {e}", file=sys.stderr)
    try:
        out = compute_move_benchmark_row(spec, stage, ckpt_path, quiet=quiet)
        if out is None:
            print("Warning: dashboard move benchmark skipped (val_move_dataset_h5 missing)", file=sys.stderr)
            return
        row, meta = out
        p = stage_benchmarks_json_path(ckpt_dir, spec["name"])
        upsert_stage_benchmarks(p, spec["name"], meta, row)
    except Exception as e:
        print(f"Warning: dashboard stage {stage} move benchmark failed: {e}", file=sys.stderr)


def refresh_dashboard_metrics_for_model(
    spec: dict[str, Any],
    *,
    stages: list[int] | None = None,
    quiet: bool = False,
    dry_run: bool = False,
) -> list[int]:
    """Recompute profile (from stage 0) and move-benchmark rows for existing checkpoints."""
    name = spec["name"]
    ckpt_dir = Path(spec["checkpoint_dir"])
    n_max = len(spec["stages"])
    to_run: list[int] = []
    for k in range(0, n_max + 1):
        if stages is not None and k not in stages:
            continue
        if stage_checkpoint_path(ckpt_dir, name, k).is_file():
            to_run.append(k)
    if dry_run:
        print(f"[dry-run] {name}: would refresh stages {to_run}", file=sys.stderr)
        return to_run
    p0 = stage_checkpoint_path(ckpt_dir, name, 0)
    if p0.is_file():
        try:
            write_profile_json(spec, p0, quiet=quiet)
        except Exception as e:
            print(f"Warning: profile failed: {e}", file=sys.stderr)
    bench_path = stage_benchmarks_json_path(ckpt_dir, name)
    for k in to_run:
        ck = stage_checkpoint_path(ckpt_dir, name, k)
        try:
            out = compute_move_benchmark_row(spec, k, ck, quiet=quiet)
            if out is None:
                print(f"Warning: move benchmark skipped for stage {k} (val H5 missing)", file=sys.stderr)
                continue
            row, meta = out
            upsert_stage_benchmarks(bench_path, name, meta, row)
            if not quiet:
                print(f"Updated dashboard metrics for {name} stage {k}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: move benchmark failed for stage {k}: {e}", file=sys.stderr)
    return to_run
