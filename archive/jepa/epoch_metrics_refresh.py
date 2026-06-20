"""Append a JSONL metrics row by re-evaluating a saved stage checkpoint (no training)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from jepa.checkpoint_paths import stage_checkpoint_path
from jepa.dashboard.scan import summarize_checkpoint
from jepa.load import load_jepa_from_checkpoint
from jepa.metrics_paths import epoch_metrics_jsonl_path
from jepa.train_stage_data import MaterializeResolutionError, resolve_materialized_loaders_for_stage
from jepa.training_loop import compute_epoch_metrics_inference


def refresh_epoch_metrics_for_stage(
    spec: dict[str, Any],
    stage: int,
    device: torch.device,
    *,
    quiet: bool = False,
    dry_run: bool = False,
) -> bool:
    """
    Load ``{name}_stage_{stage}.pt``, run one inference pass over train/val materialized data,
    append one line to ``{name}_stage_{stage}_epochs.jsonl``.

    Returns True if a row was written (or would be in dry_run), False if skipped (missing ckpt).
    """
    name = spec["name"]
    ckpt_dir = Path(spec["checkpoint_dir"])
    ckpt_path = stage_checkpoint_path(ckpt_dir, name, stage)
    if not ckpt_path.is_file():
        if not quiet:
            print(f"Warning: skip stage {stage} — missing checkpoint {ckpt_path}", file=sys.stderr)
        return False

    if dry_run:
        if not quiet:
            print(f"[dry-run] {name} stage {stage}: would refresh epoch metrics", file=sys.stderr)
        return True

    try:
        train_loader, val_loader, _train_meta = resolve_materialized_loaders_for_stage(
            spec, stage, device, quiet=quiet
        )
    except MaterializeResolutionError as e:
        print(f"Warning: {name} stage {stage}: materialize failed: {e}", file=sys.stderr)
        return False

    d = spec["defaults"]
    use_amp = bool(d.get("use_amp", True)) and device.type == "cuda"
    margin_alpha = float(d["triplet_margin_alpha"])
    vicreg_var_coef = float(d["vicreg_var_coef"])
    vicreg_std_target = float(d["vicreg_std_target"])

    model = load_jepa_from_checkpoint(ckpt_path, device=device)
    m = compute_epoch_metrics_inference(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        use_amp=use_amp,
        margin_alpha=margin_alpha,
        vicreg_var_coef=vicreg_var_coef,
        vicreg_std_target=vicreg_std_target,
    )

    summ = summarize_checkpoint(ckpt_path)
    th = (summ or {}).get("train_hparams") or {}
    best_val_saved = th.get("best_val_loss")
    best_ep_saved = th.get("best_epoch")
    epoch_row = int(best_ep_saved) if best_ep_saved is not None else 1
    best_val_so_far = float(best_val_saved) if best_val_saved is not None else float(m["val_loss"])
    best_ep_so_far = int(best_ep_saved) if best_ep_saved is not None else epoch_row

    row: dict[str, Any] = {
        "epoch": epoch_row,
        "train_loss": m["train_loss"],
        "val_loss": m["val_loss"],
        "train_pct_active": m["train_pct_active"],
        "train_mean_n_neg_within_margin": m["train_mean_n_neg_within_margin"],
        "train_pct_pos_beats_hardest_neg": m["train_pct_pos_beats_hardest_neg"],
        "train_vicreg_std_mean": m["train_vicreg_std_mean"],
        "val_pct_active": m["val_pct_active"],
        "val_mean_n_neg_within_margin": m["val_mean_n_neg_within_margin"],
        "val_pct_pos_beats_hardest_neg": m["val_pct_pos_beats_hardest_neg"],
        "val_vicreg_std_mean": m["val_vicreg_std_mean"],
        "best_val_loss_so_far": best_val_so_far,
        "best_epoch_so_far": best_ep_so_far,
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": "checkpoint_refresh",
        "refreshed_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(ckpt_path.resolve()),
        "run": {"model": name, "stage": stage},
    }

    out_path = epoch_metrics_jsonl_path(ckpt_dir, name, stage)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out_path.open("a", encoding="utf-8") as mf:
            mf.write(json.dumps(row, separators=(",", ":")) + "\n")
    except OSError as e:
        print(f"Warning: could not append {out_path}: {e}", file=sys.stderr)
        return False

    if not quiet:
        print(f"Appended checkpoint_refresh row to {out_path}", file=sys.stderr)
    return True


def resolve_refresh_device(setting: str) -> torch.device:
    if setting == "cpu":
        return torch.device("cpu")
    if setting == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def refresh_epoch_metrics_for_model(
    spec: dict[str, Any],
    *,
    device: torch.device,
    stages: list[int] | None = None,
    quiet: bool = False,
    dry_run: bool = False,
) -> list[int]:
    """
    Run :func:`refresh_epoch_metrics_for_stage` for each training stage that has a checkpoint.
    If ``stages`` is None, uses ``1 .. len(spec["stages"])`` (all training stages).
    Returns the list of stage indices that were processed successfully.
    """
    name = spec["name"]
    ckpt_dir = Path(spec["checkpoint_dir"])
    n_train = len(spec["stages"])
    want = list(stages) if stages is not None else list(range(1, n_train + 1))
    done: list[int] = []
    if dry_run:
        for stage in want:
            if stage < 1 or stage > n_train:
                continue
            p = stage_checkpoint_path(ckpt_dir, name, stage)
            if p.is_file():
                done.append(stage)
            elif not quiet:
                print(f"[dry-run] {name} stage {stage}: no checkpoint {p}", file=sys.stderr)
        return done

    for stage in want:
        if stage < 1 or stage > n_train:
            if not quiet:
                print(f"Warning: {name}: skip invalid stage {stage}", file=sys.stderr)
            continue
        p = stage_checkpoint_path(ckpt_dir, name, stage)
        if not p.is_file():
            if not quiet:
                print(f"Warning: {name}: skip stage {stage} (no {p.name})", file=sys.stderr)
            continue
        if refresh_epoch_metrics_for_stage(
            spec,
            stage,
            device,
            quiet=quiet,
            dry_run=False,
        ):
            done.append(stage)
    return done
