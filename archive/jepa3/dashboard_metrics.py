"""Dashboard JSON for jepa3: joint from+to top-1 on packed move HDF5 (no retraining)."""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from jepa3.board_masks import board_from_tensor_full, legal_to_square_mask
from jepa3.checkpoint_paths import stage_checkpoint_path
from jepa3.load import load_jepa3_from_checkpoint
from jepa3.metrics_paths import stage_benchmarks_json_path
from jepa3.packed_board_codec import packed_to_board_tensor, u64_pair_to_masks
from jepa3.packed_h5 import (
    DATASET_FROM_LEGAL_U64,
    DATASET_FROM_SQ,
    DATASET_PACKED_PRE,
    DATASET_TO_LEGAL_U64,
    DATASET_TO_SQ,
    assert_packed_h5,
)


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


def _sample_indices(n_total: int, sample_n: int, seed: int) -> np.ndarray:
    n = min(int(sample_n), n_total)
    if n <= 0:
        return np.array([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=n, replace=False))


def _resolve_benchmark_device(device_setting: str) -> torch.device:
    if device_setting == "cpu":
        return torch.device("cpu")
    if device_setting == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _masked_argmax_sq(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    x = logits.float().masked_fill(mask < 0.5, float("-inf"))
    return x.argmax(dim=-1)


def joint_from_to_top1_packed(
    model: torch.nn.Module,
    f: h5py.File,
    indices: np.ndarray,
    *,
    device: torch.device,
    use_amp: bool,
    batch_size: int,
) -> tuple[int, int, int, float]:
    """
    Fraction of positions where masked argmax(from) == label from and masked argmax(to | pred_from) == label to.

    Returns:
        n_joint_correct, n_evaluated, n_skipped, pct (0..100 on evaluated rows only; 0 if none).
    """
    n_joint = 0
    n_ev = 0
    n_skip = 0
    bs = max(1, int(batch_size))
    model.eval()

    for start in range(0, len(indices), bs):
        chunk = indices[start : start + bs]
        pre = np.asarray(f[DATASET_PACKED_PRE][chunk])
        fu = np.asarray(f[DATASET_FROM_LEGAL_U64][chunk], dtype=np.uint64)
        tu = np.asarray(f[DATASET_TO_LEGAL_U64][chunk], dtype=np.uint64)
        fs = np.asarray(f[DATASET_FROM_SQ][chunk], dtype=np.int64)
        ts = np.asarray(f[DATASET_TO_SQ][chunk], dtype=np.int64)

        b0 = pre.shape[0]
        board_b = np.zeros((b0, 8, 8, 18), dtype=np.float32)
        from_m_b = np.zeros((b0, 64), dtype=np.float32)
        to_m_stored = np.zeros((b0, 64), dtype=np.float32)
        row_ok: list[int] = []
        for i in range(b0):
            board_b[i] = packed_to_board_tensor(pre[i])
            from_m, to_m = u64_pair_to_masks(fu[i], tu[i])
            from_m_b[i] = from_m
            to_m_stored[i] = to_m
            if float(from_m.sum()) >= 0.5:
                row_ok.append(i)
            else:
                n_skip += 1

        if not row_ok:
            continue

        board_sub = board_b[row_ok]
        fm_sub = from_m_b[row_ok]
        to_stored_sub = to_m_stored[row_ok]
        fs_sub = fs[row_ok]
        ts_sub = ts[row_ok]
        b = board_sub.shape[0]

        bt = torch.from_numpy(board_sub).to(device, non_blocking=True)
        fm = torch.from_numpy(fm_sub).to(device, non_blocking=True)

        with torch.no_grad():
            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    z = model.encode_online(bt)
                    from_logits = model.forward_from_logits(z)
            else:
                z = model.encode_online(bt)
                from_logits = model.forward_from_logits(z)

            pred_from = _masked_argmax_sq(from_logits, fm)
            to_logits = model.forward_to_logits(z, pred_from)

        to_mask_t = torch.zeros(b, 64, dtype=torch.float32, device=device)
        for j in range(b):
            pf = int(pred_from[j].item())
            tf = int(fs_sub[j])
            if pf == tf:
                to_mask_t[j] = torch.from_numpy(to_stored_sub[j]).to(device, dtype=torch.float32)
            else:
                brd = board_from_tensor_full(board_sub[j])
                tm = legal_to_square_mask(brd, pf)
                to_mask_t[j] = torch.from_numpy(tm).to(device, dtype=torch.float32)

        pred_to = _masked_argmax_sq(to_logits, to_mask_t)

        for j in range(b):
            if float(to_mask_t[j].sum().item()) < 0.5:
                n_skip += 1
                continue
            n_ev += 1
            if int(pred_from[j].item()) == int(fs_sub[j]) and int(pred_to[j].item()) == int(ts_sub[j]):
                n_joint += 1

    pct = 100.0 * n_joint / n_ev if n_ev > 0 else 0.0
    return n_joint, n_ev, n_skip, pct


def compute_move_benchmark_row(
    spec: dict[str, Any],
    stage: int,
    ckpt_path: Path,
    *,
    quiet: bool = True,
    eval_batch_size: int = 256,
    sample_n_override: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """
    One stage row for ``<model>_stage_benchmarks.json``.

    ``top1_pct`` is **joint** from+to accuracy (both argmaxes match labels; ``forward_to_logits`` uses **predicted** from).
    """
    _ = quiet
    dm = spec["dashboard_metrics"]
    val_h5 = Path(spec["val_move_dataset_h5"])
    if not val_h5.is_file():
        return None
    try:
        assert_packed_h5(val_h5)
    except ValueError as e:
        raise ValueError(f"val_move_dataset_h5 must be jepa3 packed HDF5: {e}") from e

    dev = _resolve_benchmark_device(str(dm["device"]))
    use_amp = dev.type == "cuda"
    sample_n = int(sample_n_override) if sample_n_override is not None else int(dm["move_benchmark_sample_n"])
    seed = int(dm["move_benchmark_seed"])

    ckpt_path = ckpt_path.resolve()
    model = load_jepa3_from_checkpoint(ckpt_path, device=dev)

    with h5py.File(val_h5, "r") as f:
        n_total = int(f[DATASET_PACKED_PRE].shape[0])
        indices = _sample_indices(n_total, sample_n, seed)
        _nj, n_ev, n_skip, pct_val = joint_from_to_top1_packed(
            model,
            f,
            indices,
            device=dev,
            use_amp=use_amp,
            batch_size=eval_batch_size,
        )

    row: dict[str, Any] = {
        "stage": int(stage),
        "checkpoint_path": str(ckpt_path),
        "top1_pct": float(pct_val),
        "n_positions": int(n_ev),
        "n_skipped": int(n_skip),
        "benchmark": "joint_from_to_top1_packed",
    }

    train_h5 = Path(spec["train_move_dataset_h5"])
    train_ok = train_h5.is_file()
    if train_ok:
        try:
            assert_packed_h5(train_h5)
        except ValueError:
            train_ok = False
    if train_ok:
        train_seed = int(dm["move_benchmark_train_seed"])
        with h5py.File(train_h5, "r") as tf:
            n_tr = int(tf[DATASET_PACKED_PRE].shape[0])
            t_indices = _sample_indices(n_tr, sample_n, train_seed)
            _nj2, n_ev2, n_skip2, pct_tr = joint_from_to_top1_packed(
                model,
                tf,
                t_indices,
                device=dev,
                use_amp=use_amp,
                batch_size=eval_batch_size,
            )
        row["train"] = {
            "top1_pct": float(pct_tr),
            "n_positions": int(n_ev2),
            "n_skipped": int(n_skip2),
        }

    meta = {
        "val_move_dataset_h5": str(val_h5),
        "sample_n": sample_n,
        "seed": seed,
        "device_used": str(dev),
        "top1_pct_is_joint_from_to": True,
        "benchmark": "joint_from_to_top1_packed",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if train_ok:
        meta["train_move_dataset_h5"] = str(train_h5)
        meta["train_seed"] = int(dm["move_benchmark_train_seed"])
    return row, meta


def refresh_dashboard_metrics_for_model(
    spec: dict[str, Any],
    *,
    stages: list[int] | None = None,
    quiet: bool = False,
    dry_run: bool = False,
    eval_batch_size: int = 256,
    sample_n_override: int | None = None,
) -> list[int]:
    """Recompute move-benchmark rows (joint top1 only) for existing checkpoints."""
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

    bench_path = stage_benchmarks_json_path(ckpt_dir, name)
    for k in to_run:
        ck = stage_checkpoint_path(ckpt_dir, name, k)
        try:
            out = compute_move_benchmark_row(
                spec,
                k,
                ck,
                quiet=quiet,
                eval_batch_size=eval_batch_size,
                sample_n_override=sample_n_override,
            )
            if out is None:
                print(f"Warning: move benchmark skipped for stage {k} (val H5 missing)", file=sys.stderr)
                continue
            row, meta = out
            upsert_stage_benchmarks(bench_path, name, meta, row)
            if not quiet:
                print(
                    f"Updated dashboard metrics for {name} stage {k} top1_pct={row.get('top1_pct'):.2f}",
                    file=sys.stderr,
                )
        except Exception as e:
            print(f"Warning: move benchmark failed for stage {k}: {e}", file=sys.stderr)
    return to_run
