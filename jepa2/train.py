#!/usr/bin/env python3
"""
jepa2 training: streaming move HDF5, uniform legal subsampling, CE + MSE + VICReg.

  python -m jepa2.train --model NAME --stage 0     # init -> NAME_stage_0.pt
  python -m jepa2.train --model NAME --stage 1     # load stage_0, train stage 1 using stages[0]

Per-stage metrics are written once to ``metrics/{name}_stage_{N}_metrics.json`` (no per-epoch JSONL).
"""

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
from jepa2.training_loop import init_stage_zero, run_training_epochs, save_stage_checkpoint, write_stage_metrics_json
from jepa2.load import load_jepa2_from_checkpoint


def _h5_n_rows(path: Path) -> int:
    with h5py.File(path, "r") as f:
        return int(f["fen"].shape[0])


def cmd_init(spec: dict, device: torch.device) -> int:
    out = init_stage_zero(spec, device)
    print(out, file=sys.stderr)
    return 0


def cmd_train_stage(spec: dict, stage: int, device: torch.device) -> int:
    name = spec["name"]
    stages_cfg = spec["stages"]
    if stage < 1 or stage > len(stages_cfg):
        print(
            f"Error: --stage {stage} out of range; spec has {len(stages_cfg)} training stage(s) (1..{len(stages_cfg)}).",
            file=sys.stderr,
        )
        return 1

    st_idx = stage - 1
    resolved = resolve_training_config_for_stage(spec, st_idx)
    print("jepa2 resolved training config:", json.dumps(resolved, default=str, indent=2), file=sys.stderr)

    ckpt_dir = Path(spec["checkpoint_dir"])
    prev_path = stage_checkpoint_path(ckpt_dir, name, stage - 1)
    if not prev_path.is_file():
        print(f"Error: missing checkpoint {prev_path}", file=sys.stderr)
        return 1

    st = stages_cfg[st_idx]
    train_h5 = Path(spec["train_move_dataset_h5"])
    val_h5 = Path(spec["val_move_dataset_h5"])
    n_train = _h5_n_rows(train_h5)
    n_val = _h5_n_rows(val_h5)
    train_idx = sample_row_indices(n_train, int(st["sample"]["n"]), int(st["sample"]["seed"]))
    vs = spec["val_sample"]
    val_idx = sample_row_indices(n_val, int(vs["n"]), int(vs["seed"]))

    nw = int(resolved.get("dataloader_num_workers", 0))
    bs = int(resolved["train"]["batch_size"])
    train_loader = make_loader(
        train_h5,
        train_idx,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        seed=int(st["sample"]["seed"]),
    )
    val_loader = make_loader(
        val_h5,
        val_idx,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        seed=None,
    )

    model = load_jepa2_from_checkpoint(prev_path, device=device)

    run_meta = {"model": name, "stage": stage}
    best_val, best_ep, last_inf, last_train_loss, last_avg_train = run_training_epochs(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        resolved=resolved,
        metrics_run_meta=run_meta,
        global_step_seed=int(resolved.get("train_shuffle_seed", 0)),
    )

    metrics_path = stage_metrics_json_path(ckpt_dir, name, stage)
    record: dict = {
        "source": "training",
        "model": name,
        "stage": stage,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "best_val_loss": best_val,
        "best_epoch": best_ep,
        "epochs_ran": int(resolved["train"]["epochs"]),
        "train_epoch_mean_loss_last": last_train_loss,
        "ce_weight": float(resolved["ce_weight"]),
        "mse_played_weight": float(resolved["mse_played_weight"]),
        "vicreg": dict(resolved["vicreg"]),
        "metrics_path": str(metrics_path),
        "run": run_meta,
    }
    for k, v in last_avg_train.items():
        record[f"train_epoch_mean_{k}"] = v
    record.update(last_inf)
    write_stage_metrics_json(metrics_path, record)

    out_path = save_stage_checkpoint(
        model=model,
        spec=spec,
        stage=stage,
        resolved=resolved,
        train_meta={
            "storage": "streaming",
            "n_train_rows": int(train_idx.shape[0]),
            "n_val_rows": int(val_idx.shape[0]),
            "train_h5": str(train_h5),
            "val_h5": str(val_h5),
            "stage_metrics_json": str(metrics_path),
        },
        best_val=best_val,
        best_ep=best_ep,
        epochs_ran=int(resolved["train"]["epochs"]),
    )
    print(f"Saved {out_path} (best_val={best_val:.4f} @ ep {best_ep})", file=sys.stderr)
    print(f"Stage metrics: {metrics_path}", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="jepa2 staged training (streaming legals).")
    parser.add_argument("--model", type=str, required=True, help="Model name (jepa2/model_configs/{name}.yaml)")
    parser.add_argument("--stage", type=int, required=True, help="0=init; N>=1 trains stage N using stages[N-1]")
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu (default: auto)")
    args = parser.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Error: cuda requested but not available.", file=sys.stderr)
        return 1
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    try:
        spec_path = spec_path_for_model(args.model)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    spec = load_model_spec(spec_path)
    if spec["name"] != args.model:
        print(f"Error: spec name {spec['name']!r} != --model {args.model!r}", file=sys.stderr)
        return 1

    if args.stage == 0:
        return cmd_init(spec, device)
    return cmd_train_stage(spec, args.stage, device)


if __name__ == "__main__":
    raise SystemExit(main())
