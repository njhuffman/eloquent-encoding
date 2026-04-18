#!/usr/bin/env python3
"""
Chess-JEPA training from train/val move-sample HDF5s, staged checkpoints.

  python -m jepa.train --model NAME --stage 0     # init -> NAME_stage_0.pt
  python -m jepa.train --model NAME --stage 1     # load stage_0, mine+train -> stage_1.pt
  python -m jepa.train --model NAME --stage N     # load stage_{N-1}, uses spec stages[N-1]

For each training stage, JEPA tensors (boards, negatives, Elo) are built in RAM from the
move HDF5s (no derived HDF5 cache on disk). A RAM budget check runs before materialization.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from jepa.h5_bootstrap import apply_hdf5_read_safety_env

apply_hdf5_read_safety_env()

import torch

from jepa.architectures import build_model, resolve_config_for_id
from jepa.checkpoint_paths import stage_checkpoint_path
from jepa.checkpoint_utils import build_model_checkpoint
from jepa.load import load_jepa_from_checkpoint
from jepa.train_stage_data import MaterializeResolutionError, resolve_materialized_loaders_for_stage
from jepa.dashboard_metrics import run_after_stage_zero, run_after_training_stage, skip_dashboard_metrics
from jepa.metrics_paths import epoch_metrics_jsonl_path
from jepa.model_spec import load_model_spec, spec_path_for_model
from jepa.training_loop import run_training_epochs, save_submodule_sidecars


def cmd_init(spec: dict, device: torch.device) -> int:
    name = spec["name"]
    ckpt_dir = Path(spec["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out = stage_checkpoint_path(ckpt_dir, name, 0)
    arch_id = spec["architecture"]["id"]
    arch_cfg = spec["architecture"].get("config") or {}
    resolved = resolve_config_for_id(arch_id, arch_cfg)
    model = build_model(arch_id, arch_cfg).to(device)
    model.init_target_from_online()
    payload = build_model_checkpoint(
        model,
        architecture_id=arch_id,
        architecture_config=resolved,
        train_meta={"stage": 0, "init_only": True},
        train_hparams={"stage": 0, "init_only": True},
        training_spec={"name": name, "stage": 0},
    )
    torch.save(payload, out)
    save_submodule_sidecars(out, model)
    print(out, file=sys.stderr)
    if not skip_dashboard_metrics():
        run_after_stage_zero(spec, out, quiet=True)
    return 0


def cmd_train_stage(spec: dict, stage: int, device: torch.device) -> int:
    name = spec["name"]
    stages_cfg = spec["stages"]
    if stage < 1 or stage > len(stages_cfg):
        print(
            f"Error: --stage {stage} out of range; spec has stages[0..{len(stages_cfg)-1}] "
            f"for training stages 1..{len(stages_cfg)}.",
            file=sys.stderr,
        )
        return 1

    st = stages_cfg[stage - 1]
    ckpt_dir = Path(spec["checkpoint_dir"])
    arch_id = spec["architecture"]["id"]
    arch_cfg = spec["architecture"].get("config") or {}
    resolved = resolve_config_for_id(arch_id, arch_cfg)

    prev_path = stage_checkpoint_path(ckpt_dir, name, stage - 1)
    try:
        train_loader, val_loader, train_meta = resolve_materialized_loaders_for_stage(
            spec, stage, device, quiet=False
        )
    except MaterializeResolutionError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    model = load_jepa_from_checkpoint(prev_path, device=device)

    d = spec["defaults"]
    tr = st["train"]
    use_amp = bool(d.get("use_amp", True)) and device.type == "cuda"

    metrics_path = epoch_metrics_jsonl_path(ckpt_dir, name, stage)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if metrics_path.is_file():
        metrics_path.unlink()

    best_val, best_ep = run_training_epochs(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=int(tr["epochs"]),
        learning_rate=float(tr["learning_rate"]),
        weight_decay=float(tr["weight_decay"]),
        use_amp=use_amp,
        ema_momentum=float(d["ema_momentum"]),
        margin_alpha=float(d["triplet_margin_alpha"]),
        vicreg_var_coef=float(d["vicreg_var_coef"]),
        vicreg_std_target=float(d["vicreg_std_target"]),
        log_interval=int(d.get("log_interval", 100)),
        metrics_jsonl_path=metrics_path,
        metrics_run_meta={"model": name, "stage": stage},
    )

    out_path = stage_checkpoint_path(ckpt_dir, name, stage)
    spec_snap = copy.deepcopy(
        {
            "name": name,
            "stage": stage,
            "stage_config": st,
            "defaults": d,
            "architecture": spec["architecture"],
        }
    )
    payload = build_model_checkpoint(
        model,
        architecture_id=arch_id,
        architecture_config=resolved,
        train_meta=train_meta,
        train_hparams={
            "stage": stage,
            "best_val_loss": best_val,
            "best_epoch": best_ep,
            "epochs_ran": tr["epochs"],
            "learning_rate": tr["learning_rate"],
        },
        training_spec=spec_snap,
    )
    torch.save(payload, out_path)
    save_submodule_sidecars(out_path, model)
    print(f"Saved {out_path} (best_val={best_val:.4f} @ ep {best_ep})", file=sys.stderr)
    if not skip_dashboard_metrics():
        run_after_training_stage(spec, stage, out_path, quiet=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Chess-JEPA staged training from move HDF5.")
    parser.add_argument("--model", type=str, required=True, help="Model name (jepa/model_configs/{name}.yaml)")
    parser.add_argument("--stage", type=int, required=True, help="0=init only; N>=1 trains stage N using stages[N-1]")
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu (default: auto)")
    parser.add_argument(
        "--skip-dashboard-metrics",
        action="store_true",
        help="Do not write dashboard profile / val move-benchmark JSON (also: JEPA_SKIP_DASHBOARD_METRICS=1)",
    )
    args = parser.parse_args()

    if args.skip_dashboard_metrics:
        os.environ["JEPA_SKIP_DASHBOARD_METRICS"] = "1"

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
