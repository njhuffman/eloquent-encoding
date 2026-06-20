"""Resolve materialized train/val arrays and DataLoaders for a training stage (shared by train + refresh)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import h5py
import psutil
import torch
from torch.utils.data import DataLoader

from jepa.checkpoint_paths import stage_checkpoint_path
from jepa.dataset import get_dataloaders_from_arrays
from jepa.load import load_jepa_from_checkpoint
from jepa.materialize import (
    estimate_materialized_split_bytes,
    materialize_jepa_split,
    train_pool_indices,
    val_indices,
)

MINING_POSITION_BATCH = 64
_MATERIALIZE_RAM_FRACTION = 0.5


class MaterializeResolutionError(RuntimeError):
    """Failed to resolve or build materialized data for a stage."""


def _move_h5_length(path: Path) -> int:
    with h5py.File(path, "r") as f:
        return int(f["fen"].shape[0])


def _assert_split_k_matches(
    negs: Any, path_label: str, expected_k: int
) -> None:
    k = int(negs.shape[1])
    if k != expected_k:
        raise ValueError(
            f"{path_label}: negatives have K={k} but this stage expects K={expected_k} "
            "(n_hard + m_random). Match hard_negatives in the spec."
        )


def _check_ram_budget(*, n_train_rows: int, n_val_rows: int, k_neg: int) -> None:
    need = estimate_materialized_split_bytes(n_train_rows, k_neg) + estimate_materialized_split_bytes(
        n_val_rows, k_neg
    )
    avail = int(psutil.virtual_memory().available)
    cap = int(_MATERIALIZE_RAM_FRACTION * avail)
    if need > cap:
        need_gib = need / (1024**3)
        cap_gib = cap / (1024**3)
        avail_gib = avail / (1024**3)
        raise MaterializeResolutionError(
            f"Materialized JEPA tensors need ~{need_gib:.2f} GiB (upper bound for {n_train_rows} train + "
            f"{n_val_rows} val rows at K={k_neg}), but the allowed budget is ~{cap_gib:.2f} GiB "
            f"({_MATERIALIZE_RAM_FRACTION:.0%} of ~{avail_gib:.2f} GiB available RAM). "
            "Reduce sample sizes (stages[].sample.n, val_sample.n) or free memory."
        )


def resolve_materialized_loaders_for_stage(
    spec: dict[str, Any],
    stage: int,
    device: torch.device,
    *,
    quiet: bool = False,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    """
    Build train/val DataLoaders for ``stage`` (1-based training stage).

    Raises MaterializeResolutionError on missing inputs, bad config, empty materialization, or RAM budget.
    """
    _ = quiet
    name = spec["name"]
    stages_cfg = spec["stages"]
    if stage < 1 or stage > len(stages_cfg):
        raise MaterializeResolutionError(
            f"stage {stage} out of range; spec has training stages 1..{len(stages_cfg)}"
        )

    st = stages_cfg[stage - 1]
    ckpt_dir = Path(spec["checkpoint_dir"])

    prev_path = stage_checkpoint_path(ckpt_dir, name, stage - 1)
    if not prev_path.is_file():
        raise MaterializeResolutionError(f"missing checkpoint {prev_path} (need stage {stage - 1} first)")

    train_move_h5 = Path(spec["train_move_dataset_h5"])
    val_move_h5 = Path(spec["val_move_dataset_h5"])
    for label, p in (("train_move_dataset_h5", train_move_h5), ("val_move_dataset_h5", val_move_h5)):
        if not p.is_file():
            raise MaterializeResolutionError(f"{label} not found: {p}")

    n_train_rows = _move_h5_length(train_move_h5)
    n_val_rows = _move_h5_length(val_move_h5)
    vs = spec["val_sample"]
    val_idx = val_indices(n_val_rows, vs["n"], vs["seed"])
    train_idx = train_pool_indices(
        n_train_rows,
        set(),
        st["sample"]["n"],
        st["sample"]["seed"],
    )

    n_train_cap = len(train_idx)
    n_val_cap = len(val_idx)

    hn = st["hard_negatives"]
    k_neg = int(hn["n_hard"] + hn["m_random"])

    evaluate_legals_n = hn.get("evaluate_legals_n")
    if evaluate_legals_n is not None and evaluate_legals_n <= k_neg:
        raise MaterializeResolutionError(
            f"hard_negatives.evaluate_legals_n ({evaluate_legals_n}) must be > "
            f"n_hard + m_random ({k_neg})"
        )

    use_hard = stage >= 2
    _check_ram_budget(n_train_rows=n_train_cap, n_val_rows=n_val_cap, k_neg=k_neg)

    mine_model = None
    if use_hard:
        mine_model = load_jepa_from_checkpoint(prev_path, device=device)
        mine_model.eval()

    if not quiet:
        print(
            f"Materializing train ({len(train_idx)} rows), use_hard_mining={use_hard}...",
            file=sys.stderr,
        )
    rep_tr, arr_tr = materialize_jepa_split(
        train_move_h5,
        train_idx,
        progress_desc="materialize train",
        model=mine_model,
        device=device,
        k_neg=k_neg,
        n_hard=hn["n_hard"],
        m_random=hn["m_random"],
        use_hard_mining=use_hard,
        neg_seed=st["sample"]["seed"] + 17,
        mining_position_batch=MINING_POSITION_BATCH,
        evaluate_legals_n=evaluate_legals_n,
    )
    if not quiet:
        print(f"Train report: {rep_tr}", file=sys.stderr)

    if not quiet:
        print("Materializing val...", file=sys.stderr)
    rep_va, arr_va = materialize_jepa_split(
        val_move_h5,
        val_idx,
        progress_desc="materialize val",
        model=mine_model,
        device=device,
        k_neg=k_neg,
        n_hard=hn["n_hard"],
        m_random=hn["m_random"],
        use_hard_mining=use_hard,
        neg_seed=vs["seed"] + 99,
        mining_position_batch=MINING_POSITION_BATCH,
        evaluate_legals_n=evaluate_legals_n,
    )
    if not quiet:
        print(f"Val report: {rep_va}", file=sys.stderr)

    if rep_tr["n_written"] == 0:
        raise MaterializeResolutionError("no training rows materialized")
    if rep_va["n_written"] == 0:
        raise MaterializeResolutionError("no val rows materialized")

    _bt_tr, _pos_tr, negs_tr, _elo_tr = arr_tr
    _bt_va, _pos_va, negs_va, _elo_va = arr_va
    _assert_split_k_matches(negs_tr, "train", k_neg)
    _assert_split_k_matches(negs_va, "val", k_neg)

    d = spec["defaults"]
    tr = st["train"]
    workers = int(d.get("dataloader_num_workers", 0))

    train_loader, val_loader = get_dataloaders_from_arrays(
        arr_tr,
        arr_va,
        batch_size=int(tr["batch_size"]),
        num_workers=workers,
    )

    n_train = int(arr_tr[0].shape[0])
    n_val = int(arr_va[0].shape[0])
    train_meta: dict[str, Any] = {
        "stage": stage,
        "n_train_boards": n_train,
        "n_val_boards": n_val,
        "train_move_dataset_h5": str(train_move_h5),
        "val_move_dataset_h5": str(val_move_h5),
        "materialized": "ram",
        "train_materialize_report": rep_tr,
        "val_materialize_report": rep_va,
    }

    return train_loader, val_loader, train_meta
