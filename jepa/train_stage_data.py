"""Resolve materialized train/val HDF5s and DataLoaders for a training stage (shared by train + refresh)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import h5py
import torch
from torch.utils.data import DataLoader

from jepa.architectures import resolve_config_for_id
from jepa.checkpoint_paths import stage_checkpoint_path
from jepa.dataset import assert_h5_k_matches, get_dataloaders, h5_transition_counts
from jepa.load import load_jepa_from_checkpoint
from jepa.materialize import materialize_jepa_split, train_pool_indices, val_indices
from jepa.materialize_cache import (
    build_materialize_cache_key,
    file_fingerprint,
    hash_materialize_cache_key,
    materialize_cache_paths,
    try_load_materialize_cache,
    write_materialize_cache_manifest,
)

MINING_POSITION_BATCH = 64


class MaterializeResolutionError(RuntimeError):
    """Failed to resolve or build materialized HDF5s for a stage."""


def _move_h5_length(path: Path) -> int:
    with h5py.File(path, "r") as f:
        return int(f["fen"].shape[0])


def resolve_materialized_loaders_for_stage(
    spec: dict[str, Any],
    stage: int,
    device: torch.device,
    *,
    rematerialize: bool,
    quiet: bool = False,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    """
    Build the same train/val DataLoaders as ``jepa.train`` for ``stage`` (1-based training stage).

    Raises MaterializeResolutionError on missing inputs, bad config, or empty materialization.
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
    cache_dir = Path(spec["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

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

    arch_id = spec["architecture"]["id"]
    arch_cfg = spec["architecture"].get("config") or {}
    resolved = resolve_config_for_id(arch_id, arch_cfg)
    hn = st["hard_negatives"]
    k_neg = int(hn["n_hard"] + hn["m_random"])

    evaluate_legals_n = hn.get("evaluate_legals_n")
    if evaluate_legals_n is not None and evaluate_legals_n <= k_neg:
        raise MaterializeResolutionError(
            f"hard_negatives.evaluate_legals_n ({evaluate_legals_n}) must be > "
            f"n_hard + m_random ({k_neg})"
        )

    use_hard = stage >= 2
    train_fp = file_fingerprint(train_move_h5)
    val_fp = file_fingerprint(val_move_h5)
    mine_ckpt_fp = file_fingerprint(prev_path) if use_hard else None
    cache_key = build_materialize_cache_key(
        stage=stage,
        use_hard_mining=use_hard,
        architecture_id=arch_id,
        architecture_resolved=resolved,
        k_neg=k_neg,
        n_hard=hn["n_hard"],
        m_random=hn["m_random"],
        mining_position_batch=MINING_POSITION_BATCH,
        train_neg_seed=st["sample"]["seed"] + 17,
        val_neg_seed=vs["seed"] + 99,
        n_train_rows=n_train_rows,
        n_val_rows=n_val_rows,
        sample_n=st["sample"]["n"],
        sample_seed=st["sample"]["seed"],
        val_sample_n=vs["n"],
        val_sample_seed=vs["seed"],
        train_move_fingerprint=train_fp,
        val_move_fingerprint=val_fp,
        mining_checkpoint_fingerprint=mine_ckpt_fp,
        evaluate_legals_n=evaluate_legals_n,
    )
    key_sha256 = hash_materialize_cache_key(cache_key)
    train_h5_path, val_h5_path, manifest_path = materialize_cache_paths(
        cache_dir, name, stage, key_sha256
    )

    rep_tr: dict[str, Any]
    rep_va: dict[str, Any]
    if not rematerialize:
        loaded = try_load_materialize_cache(
            cache_key=cache_key,
            key_sha256=key_sha256,
            train_h5_path=train_h5_path,
            val_h5_path=val_h5_path,
            manifest_path=manifest_path,
            k_neg=k_neg,
        )
        if loaded is not None:
            rep_tr, rep_va = loaded
            if not quiet:
                print(
                    f"Reusing materialized cache {train_h5_path.name} / {val_h5_path.name}",
                    file=sys.stderr,
                )
        else:
            rep_tr = {}
            rep_va = {}
    else:
        rep_tr = {}
        rep_va = {}

    if not rep_tr:
        mine_model = None
        if use_hard:
            mine_model = load_jepa_from_checkpoint(prev_path, device=device)
            mine_model.eval()

        if not quiet:
            print(
                f"Materializing train ({len(train_idx)} rows), use_hard_mining={use_hard}...",
                file=sys.stderr,
            )
        rep_tr = materialize_jepa_split(
            train_move_h5,
            train_idx,
            train_h5_path,
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
        rep_va = materialize_jepa_split(
            val_move_h5,
            val_idx,
            val_h5_path,
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

        write_materialize_cache_manifest(
            manifest_path,
            cache_key=cache_key,
            key_sha256=key_sha256,
            train_h5_path=train_h5_path,
            val_h5_path=val_h5_path,
        )

    if rep_tr["n_written"] == 0:
        raise MaterializeResolutionError("no training rows materialized")
    if rep_va["n_written"] == 0:
        raise MaterializeResolutionError("no val rows materialized")

    assert_h5_k_matches(train_h5_path, k_neg)
    assert_h5_k_matches(val_h5_path, k_neg)

    d = spec["defaults"]
    tr = st["train"]
    workers = int(d.get("dataloader_num_workers", 0))

    train_loader, val_loader = get_dataloaders(
        train_h5_path,
        val_h5_path,
        batch_size=int(tr["batch_size"]),
        num_workers=workers,
        in_memory=bool(d.get("in_memory", False)),
    )

    n_train, n_val = h5_transition_counts(train_h5_path, val_h5_path)
    train_meta: dict[str, Any] = {
        "stage": stage,
        "n_train_boards": n_train,
        "n_val_boards": n_val,
        "train_move_dataset_h5": str(train_move_h5),
        "val_move_dataset_h5": str(val_move_h5),
        "train_h5": str(train_h5_path),
        "val_h5": str(val_h5_path),
        "train_materialize_report": rep_tr,
        "val_materialize_report": rep_va,
    }

    return train_loader, val_loader, train_meta
