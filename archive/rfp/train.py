#!/usr/bin/env python3
"""
rfp staged training (YAML spec), jepa3-style stages:

  python -m rfp.train --model NAME --stage 0     # init trainable blocks -> NAME_stage_0.pt
  python -m rfp.train --model NAME --stage 1     # load stage_0, train with stages[0] -> stage_1.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jepa.h5_bootstrap import apply_hdf5_read_safety_env

apply_hdf5_read_safety_env()

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from rfp.checkpoint_paths import stage_checkpoint_path
from rfp.dataset import make_rfp_loader, sample_row_indices
from rfp.h5_io import rfp_h5_attrs, rfp_h5_row_count
from rfp.metrics_paths import stage_metrics_json_path, write_stage_metrics_json
from rfp.model import ResidualFromPredictor, load_gfp_head_from_checkpoint
from rfp.model_spec import load_model_spec, resolve_training_config_for_stage, spec_path_for_model
from rfp.training_loop import run_rfp_training_epochs


def _assert_h5_matches_spec(spec: dict[str, Any]) -> tuple[int, int]:
    train_h5 = Path(spec["train_dataset_h5"])
    val_h5 = Path(spec["val_dataset_h5"])
    hl_tr, dm_tr = rfp_h5_attrs(train_h5)
    hl_va, dm_va = rfp_h5_attrs(val_h5)
    if (hl_tr, dm_tr) != (hl_va, dm_va):
        raise ValueError(
            f"train/val rfp HDF5 attrs mismatch: train (history_len={hl_tr}, d_model={dm_tr}) "
            f"vs val (history_len={hl_va}, d_model={dm_va})"
        )
    cfg = spec["architecture"]["config"]
    if int(cfg["history_len"]) != hl_tr:
        raise ValueError(
            f"HDF5 history_len={hl_tr} != spec architecture.history_len={cfg['history_len']}"
        )
    return hl_tr, dm_tr


def _build_model(spec: dict[str, Any], device: torch.device, d_model: int) -> ResidualFromPredictor:
    cfg = spec["architecture"]["config"]
    gfp_head = load_gfp_head_from_checkpoint(
        spec["gfp_checkpoint"],
        d_model=d_model,
        device=device,
    )
    return ResidualFromPredictor(
        gfp_head,
        history_len=int(cfg["history_len"]),
        d_model=int(d_model),
        mixer_dim=int(cfg["mixer_dim"]),
        mixer_depth=int(cfg["mixer_depth"]),
        mixer_tokens_mlp_dim=int(cfg["mixer_tokens_mlp_dim"]),
        mixer_channels_mlp_dim=int(cfg["mixer_channels_mlp_dim"]),
        mixer_dropout=float(cfg["mixer_dropout"]),
        elo_num_buckets=int(cfg["elo_num_buckets"]),
        elo_embed_dim=int(cfg["elo_embed_dim"]),
        residual_hidden=int(cfg["residual_hidden"]),
        residual_depth=int(cfg["residual_depth"]),
    ).to(device)


def _save_stage_checkpoint(
    path: Path,
    *,
    model: ResidualFromPredictor,
    spec: dict[str, Any],
    stage: int,
    resolved: dict[str, Any],
    train_meta: dict[str, Any],
    best_val_loss: float,
    best_epoch: int,
    epochs_ran: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mixer_state_dict": model.mixer.state_dict(),
            "elo_embed_state_dict": model.elo_embed.state_dict(),
            "residual_mlp_state_dict": model.residual_mlp.state_dict(),
            "model_name": spec["name"],
            "stage": int(stage),
            "spec_path": spec.get("_spec_path", ""),
            "encoder_checkpoint": spec["encoder_checkpoint"],
            "gfp_checkpoint": spec["gfp_checkpoint"],
            "architecture": spec["architecture"],
            "resolved_summary": {
                "train": dict(resolved["train"]),
                "sq_ce_label_smoothing": float(resolved["sq_ce_label_smoothing"]),
                "elo_null_prob": float(resolved["elo_null_prob"]),
            },
            "train_meta": train_meta,
            "best_val_loss": float(best_val_loss),
            "best_epoch": int(best_epoch),
            "epochs_ran": int(epochs_ran),
        },
        path,
    )


def _load_trainable_from_checkpoint(model: ResidualFromPredictor, ckpt_path: Path, device: torch.device) -> dict:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    for key in ("mixer_state_dict", "elo_embed_state_dict", "residual_mlp_state_dict"):
        if key not in ck:
            raise KeyError(f"missing {key} in {ckpt_path}")
    model.mixer.load_state_dict(ck["mixer_state_dict"], strict=True)
    model.elo_embed.load_state_dict(ck["elo_embed_state_dict"], strict=True)
    model.residual_mlp.load_state_dict(ck["residual_mlp_state_dict"], strict=True)
    return ck


def _assert_ckpt_matches_spec(ck: dict[str, Any], spec: dict[str, Any]) -> None:
    if ck.get("model_name") != spec["name"]:
        raise ValueError(
            f"checkpoint model_name {ck.get('model_name')!r} != spec name {spec['name']!r}"
        )
    if ck.get("architecture") != spec["architecture"]:
        raise ValueError("checkpoint architecture != current spec architecture")


def _h5_n_rows(path: Path) -> int:
    return rfp_h5_row_count(path)


def cmd_init(spec: dict[str, Any], device: torch.device) -> int:
    torch.manual_seed(int(spec["defaults"]["seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(spec["defaults"]["seed"]))

    out = stage_checkpoint_path(spec["checkpoint_dir"], spec["name"], 0)
    if out.is_file():
        print(f"Refusing to overwrite existing {out}", file=sys.stderr)
        return 1

    hl, d_model = _assert_h5_matches_spec(spec)
    _ = hl
    model = _build_model(spec, device, d_model)
    _save_stage_checkpoint(
        out,
        model=model,
        spec=spec,
        stage=0,
        resolved={"train": {}, "sq_ce_label_smoothing": 0.0, "elo_null_prob": 0.0},
        train_meta={"init": True},
        best_val_loss=float("nan"),
        best_epoch=0,
        epochs_ran=0,
    )
    print(out, file=sys.stderr)
    return 0


def cmd_train_stage(spec: dict[str, Any], stage: int, device: torch.device) -> int:
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
    print("rfp resolved training config:", json.dumps(resolved, default=str, indent=2), file=sys.stderr)

    ckpt_dir = Path(spec["checkpoint_dir"])
    prev_path = stage_checkpoint_path(ckpt_dir, name, stage - 1)
    if not prev_path.is_file():
        print(f"Error: missing checkpoint {prev_path}", file=sys.stderr)
        return 1

    torch.manual_seed(int(spec["defaults"]["seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(spec["defaults"]["seed"]))

    hl, d_model = _assert_h5_matches_spec(spec)
    _ = hl
    model = _build_model(spec, device, d_model)
    ck = _load_trainable_from_checkpoint(model, prev_path, device)
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
    train_loader = make_rfp_loader(
        train_h5,
        train_idx,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        seed=int(st["sample"]["seed"]),
    )
    val_loader = make_rfp_loader(
        val_h5,
        val_idx,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        seed=None,
    )

    run_meta = {"model": name, "stage": stage}
    best_val, best_ep, last_val, last_train_loss, last_avg_train, epochs_ran, early_stopped = (
        run_rfp_training_epochs(
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
    record: dict[str, Any] = {
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
        "elo_null_prob": float(resolved["elo_null_prob"]),
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
        "storage": "rfp_hdf5",
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
    parser = argparse.ArgumentParser(description="rfp staged training (YAML spec, jepa3-style).")
    parser.add_argument("--model", type=str, required=True, help="Model name (rfp/model_configs/{name}.yaml)")
    parser.add_argument("--stage", type=int, required=True, help="0=init trainable blocks; N>=1 trains stage N")
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
