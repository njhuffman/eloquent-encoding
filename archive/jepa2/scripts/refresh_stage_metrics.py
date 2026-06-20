#!/usr/bin/env python3
"""Recompute and overwrite jepa2 per-stage metrics JSON (streaming val/train, no epoch JSONL)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from jepa.h5_bootstrap import apply_hdf5_read_safety_env

apply_hdf5_read_safety_env()

import h5py
import torch

from jepa2.checkpoint_paths import stage_checkpoint_path
from jepa2.dataset import make_loader, sample_row_indices
from jepa2.metrics_paths import stage_metrics_json_path
from jepa2.model_spec import load_model_spec, resolve_training_config_for_stage, spec_path_for_model
from jepa2.training_loop import compute_epoch_metrics_inference, write_stage_metrics_json
from jepa2.load import load_checkpoint_mapping, load_jepa2_from_checkpoint


def _h5_n_rows(path: Path) -> int:
    with h5py.File(path, "r") as f:
        return int(f["fen"].shape[0])


def main() -> int:
    p = argparse.ArgumentParser(description="Refresh jepa2 stage metrics JSON from checkpoint.")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--stage", type=int, required=True, help="Training stage (>=1)")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    spec_path = spec_path_for_model(args.model)
    spec = load_model_spec(spec_path)
    stage = int(args.stage)
    if stage < 1 or stage > len(spec["stages"]):
        print("Error: invalid stage", file=sys.stderr)
        return 1

    ckpt_path = stage_checkpoint_path(Path(spec["checkpoint_dir"]), spec["name"], stage)
    ckpt = load_checkpoint_mapping(ckpt_path, map_location="cpu")
    resolved = resolve_training_config_for_stage(spec, stage - 1)
    legacy = ckpt.get("resolved_training") or (ckpt.get("extra") or {}).get("resolved_training")
    if isinstance(legacy, dict) and "mse_played_weight" in legacy:
        raise ValueError(
            "Checkpoint was saved with deprecated 'mse_played_weight'. "
            "Retrain from an updated spec (use vicreg.inv_coef) or remove resolved_training from the checkpoint."
        )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_jepa2_from_checkpoint(ckpt_path, device=device)

    st = spec["stages"][stage - 1]
    train_h5 = Path(spec["train_move_dataset_h5"])
    val_h5 = Path(spec["val_move_dataset_h5"])
    train_idx = sample_row_indices(_h5_n_rows(train_h5), int(st["sample"]["n"]), int(st["sample"]["seed"]))
    vs = spec["val_sample"]
    val_idx = sample_row_indices(_h5_n_rows(val_h5), int(vs["n"]), int(vs["seed"]))
    bs = int(resolved["train"]["batch_size"])
    nw = int(resolved.get("dataloader_num_workers", 0))
    train_loader = make_loader(train_h5, train_idx, batch_size=bs, shuffle=False, num_workers=nw, seed=None)
    val_loader = make_loader(val_h5, val_idx, batch_size=bs, shuffle=False, num_workers=nw, seed=None)

    use_amp = bool(resolved.get("use_amp", True)) and device.type == "cuda"
    th = ckpt.get("train_hparams") or {}
    inf = compute_epoch_metrics_inference(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        resolved=resolved,
        use_amp=use_amp,
        val_seed=int(resolved.get("val_legal_seed", 42)),
        epoch=int(th.get("best_epoch", 0) or 0),
    )

    metrics_path = stage_metrics_json_path(Path(spec["checkpoint_dir"]), spec["name"], stage)
    record = {
        "source": "checkpoint_refresh",
        "model": spec["name"],
        "stage": stage,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(ckpt_path),
        "best_val_loss": float(th.get("best_val_loss", float("nan"))),
        "best_epoch": int(th.get("best_epoch", 0) or 0),
        "epochs_ran": int(th.get("epochs_ran", 0) or 0),
        "ce_weight": float(resolved["ce_weight"]),
        "vicreg": dict(resolved["vicreg"]),
        "metrics_path": str(metrics_path),
    }
    record.update(inf)
    written = write_stage_metrics_json(metrics_path, record)
    print(written, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
