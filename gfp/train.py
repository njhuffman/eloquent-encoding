#!/usr/bin/env python3
"""
gfp staged training (YAML spec), jepa3-style stages:

  python -m gfp.train --model NAME --stage 0     # init random head -> NAME_stage_0.pt
  python -m gfp.train --model NAME --stage 1     # load stage_0, train with stages[0] -> stage_1.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from jepa.h5_bootstrap import apply_hdf5_read_safety_env

apply_hdf5_read_safety_env()

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from gfp.checkpoint_paths import stage_checkpoint_path
from gfp.dataset import make_gfp_loader, sample_row_indices
from gfp.encoder import load_jepa3_encoder_from_checkpoint
from gfp.h5_io import gfp_h5_row_count
from gfp.metrics_paths import stage_metrics_json_path, write_stage_metrics_json
from gfp.model import GlobalFromPredictor
from gfp.model_spec import load_model_spec, resolve_training_config_for_stage, spec_path_for_model
from gfp.training_loop import run_gfp_training_epochs


def _build_model(spec: dict, device: torch.device) -> GlobalFromPredictor:
    strict = bool(spec["defaults"]["encoder_strict"])
    encoder = load_jepa3_encoder_from_checkpoint(
        spec["encoder_checkpoint"],
        device=device,
        strict=strict,
    )
    cfg = spec["architecture"]["config"]
    return GlobalFromPredictor(
        encoder,
        head_hidden=int(cfg["head_hidden"]),
        head_depth=int(cfg["head_depth"]),
    ).to(device)


def _save_stage_checkpoint(
    path: Path,
    *,
    model: GlobalFromPredictor,
    spec: dict,
    stage: int,
    resolved: dict[str, Any],
    train_meta: dict,
    best_val_loss: float,
    best_epoch: int,
    epochs_ran: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_state_dict": model.head.state_dict(),
            "model_name": spec["name"],
            "stage": int(stage),
            "spec_path": spec.get("_spec_path", ""),
            "encoder_checkpoint": spec["encoder_checkpoint"],
            "architecture": spec["architecture"],
            "resolved_summary": {
                "train": dict(resolved["train"]),
                "sq_ce_label_smoothing": float(resolved["sq_ce_label_smoothing"]),
            },
            "train_meta": train_meta,
            "best_val_loss": float(best_val_loss),
            "best_epoch": int(best_epoch),
            "epochs_ran": int(epochs_ran),
        },
        path,
    )


def _load_head_from_checkpoint(model: GlobalFromPredictor, ckpt_path: Path, device: torch.device) -> dict:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "head_state_dict" not in ck:
        raise KeyError(f"missing head_state_dict in {ckpt_path}")
    model.head.load_state_dict(ck["head_state_dict"], strict=True)
    return ck


def _assert_ckpt_matches_spec(ck: dict, spec: dict) -> None:
    if ck.get("model_name") != spec["name"]:
        raise ValueError(
            f"checkpoint model_name {ck.get('model_name')!r} != spec name {spec['name']!r}"
        )
    if ck.get("architecture") != spec["architecture"]:
        raise ValueError("checkpoint architecture != current spec architecture")


def _h5_n_rows(path: Path) -> int:
    return gfp_h5_row_count(path)


def cmd_init(spec: dict, device: torch.device) -> int:
    torch.manual_seed(int(spec["defaults"]["seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(spec["defaults"]["seed"]))

    out = stage_checkpoint_path(spec["checkpoint_dir"], spec["name"], 0)
    if out.is_file():
        print(f"Refusing to overwrite existing {out}", file=sys.stderr)
        return 1

    model = _build_model(spec, device)
    _save_stage_checkpoint(
        out,
        model=model,
        spec=spec,
        stage=0,
        resolved={"train": {}, "sq_ce_label_smoothing": 0.0},
        train_meta={"init": True},
        best_val_loss=float("nan"),
        best_epoch=0,
        epochs_ran=0,
    )
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
    print("gfp resolved training config:", json.dumps(resolved, default=str, indent=2), file=sys.stderr)

    ckpt_dir = Path(spec["checkpoint_dir"])
    prev_path = stage_checkpoint_path(ckpt_dir, name, stage - 1)
    if not prev_path.is_file():
        print(f"Error: missing checkpoint {prev_path}", file=sys.stderr)
        return 1

    torch.manual_seed(int(spec["defaults"]["seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(spec["defaults"]["seed"]))

    model = _build_model(spec, device)
    ck = _load_head_from_checkpoint(model, prev_path, device)
    _assert_ckpt_matches_spec(ck, spec)

    st = stages_cfg[st_idx]
    train_h5 = Path(spec["train_dataset_h5"])
    val_h5 = Path(spec["val_dataset_h5"])
    n_train = _h5_n_rows(train_h5)
    n_val = _h5_n_rows(val_h5)
    train_idx = sample_row_indices(n_train, int(st["sample"]["n"]), int(st["sample"]["seed"]))
    vs = spec["val_sample"]
    val_idx = sample_row_indices(n_val, int(vs["n"]), int(vs["seed"]))

    nw = int(resolved.get("dataloader_num_workers", 0))
    bs = int(resolved["train"]["batch_size"])
    train_loader = make_gfp_loader(
        train_h5,
        train_idx,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        seed=int(st["sample"]["seed"]),
    )
    val_loader = make_gfp_loader(
        val_h5,
        val_idx,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        seed=None,
    )

    run_meta = {"model": name, "stage": stage}
    best_val, best_ep, last_val, last_train_loss, last_avg_train, epochs_ran, early_stopped = (
        run_gfp_training_epochs(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            resolved=resolved,
            metrics_run_meta=run_meta,
        )
    )

    metrics_path = stage_metrics_json_path(ckpt_dir, name, stage)
    sched_epochs = int(resolved["train"]["epochs"])
    record: dict = {
        "source": "training",
        "model": name,
        "stage": stage,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "best_val_loss": best_val,
        "best_epoch": best_ep,
        "epochs_ran": epochs_ran,
        "train_epochs_scheduled": sched_epochs,
        "early_stopped": early_stopped,
        "train_epoch_mean_loss_last": last_train_loss,
        "sq_ce_label_smoothing": float(resolved["sq_ce_label_smoothing"]),
        "metrics_path": str(metrics_path),
        "run": run_meta,
    }
    for k, v in last_avg_train.items():
        record[f"train_epoch_mean_{k}"] = v
    record.update({f"val_{k}": v for k, v in last_val.items()})
    est = resolved.get("early_stop_train_top1")
    if est is not None:
        record["early_stop_train_top1_threshold"] = float(est)
    metrics_written = write_stage_metrics_json(metrics_path, record)
    record["metrics_path"] = str(metrics_written)

    out_path = stage_checkpoint_path(ckpt_dir, name, stage)
    train_meta = {
        "storage": "gfp_hdf5",
        "n_train_rows": int(train_idx.shape[0]),
        "n_val_rows": int(val_idx.shape[0]),
        "train_h5": str(train_h5),
        "val_h5": str(val_h5),
        "stage_metrics_json": str(metrics_written),
        "prev_checkpoint": str(prev_path),
    }
    _save_stage_checkpoint(
        out_path,
        model=model,
        spec=spec,
        stage=stage,
        resolved=resolved,
        train_meta=train_meta,
        best_val_loss=best_val,
        best_epoch=best_ep,
        epochs_ran=epochs_ran,
    )
    print(f"Saved {out_path} (best_val_loss={best_val:.4f} @ ep {best_ep})", file=sys.stderr)
    print(f"Stage metrics: {metrics_written}", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="gfp staged training (YAML spec, jepa3-style).")
    parser.add_argument("--model", type=str, required=True, help="Model name (gfp/model_configs/{name}.yaml)")
    parser.add_argument("--stage", type=int, required=True, help="0=init head; N>=1 trains stage N using stages[N-1]")
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
    spec["_spec_path"] = str(spec_path.resolve())
    if spec["name"] != args.model:
        print(f"Error: spec name {spec['name']!r} != --model {args.model!r}", file=sys.stderr)
        return 1

    if args.stage == 0:
        return cmd_init(spec, device)
    return cmd_train_stage(spec, args.stage, device)


if __name__ == "__main__":
    raise SystemExit(main())
