"""CPU profile + move-ranking benchmarks for jepa2 (shared prediction_benchmark with jepa2 loader)."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import torch

from jepa.config import BOARD_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH
from jepa.dashboard_metrics import (
    _assert_move_h5_schema,
    _json_sanitize,
    upsert_stage_benchmarks,
)
from jepa.scripts.prediction_benchmark import _sample_indices, run_move_and_positive_metrics_for_checkpoint

from jepa2.checkpoint_paths import stage_checkpoint_path
from jepa2.load import load_jepa2_from_checkpoint
from jepa2.metrics_paths import model_profile_json_path, stage_benchmarks_json_path


def _jepa2_load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    return load_jepa2_from_checkpoint(ckpt_path, device=device)


def profile_cpu_single_forward(ckpt_path: Path, *, warmup: int = 3, repeats: int = 20) -> dict[str, Any]:
    device = torch.device("cpu")
    model = load_jepa2_from_checkpoint(ckpt_path, device=device)
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


def _resolve_benchmark_device(device_setting: str) -> torch.device:
    if device_setting == "cpu":
        return torch.device("cpu")
    if device_setting == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    dev = _resolve_benchmark_device(str(dm["device"]))
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
            load_model=_jepa2_load_model,
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
                load_model=_jepa2_load_model,
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
