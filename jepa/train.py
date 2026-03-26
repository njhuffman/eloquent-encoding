#!/usr/bin/env python3
"""
Chess-JEPA training from train/val move-sample HDF5s, staged checkpoints.

  python -m jepa.train --model NAME --stage 0     # init -> NAME_stage_0.pt
  python -m jepa.train --model NAME --stage 1     # load stage_0, mine+train -> stage_1.pt
  python -m jepa.train --model NAME --stage N     # load stage_{N-1}, uses spec stages[N-1]

Materialized train/val HDF5s (mining / negatives) are cached under cache_dir with a
content hash (seeds, move files, architecture, prior checkpoint when hard-mining).
Changing learning rate or epochs reuses that cache; use --rematerialize to rebuild.
Old unkeyed files named {name}_stage{N}_train.h5 are no longer written — safe to delete.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from jepa.h5_bootstrap import apply_hdf5_read_safety_env

apply_hdf5_read_safety_env()

import h5py
import torch

from jepa.architectures import build_model, resolve_config_for_id
from jepa.checkpoint_paths import stage_checkpoint_path
from jepa.checkpoint_utils import build_model_checkpoint
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
from jepa.model_spec import load_model_spec, spec_path_for_model
from jepa.training_loop import run_training_epochs, save_submodule_sidecars


def _move_h5_length(path: Path) -> int:
    with h5py.File(path, "r") as f:
        return int(f["fen"].shape[0])


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
    return 0


MINING_POSITION_BATCH = 64


def cmd_train_stage(spec: dict, stage: int, device: torch.device, *, rematerialize: bool) -> int:
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
    cache_dir = Path(spec["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    prev_path = stage_checkpoint_path(ckpt_dir, name, stage - 1)
    if not prev_path.is_file():
        print(f"Error: missing checkpoint {prev_path} (need stage {stage - 1} first).", file=sys.stderr)
        return 1

    train_move_h5 = Path(spec["train_move_dataset_h5"])
    val_move_h5 = Path(spec["val_move_dataset_h5"])
    for label, p in (("train_move_dataset_h5", train_move_h5), ("val_move_dataset_h5", val_move_h5)):
        if not p.is_file():
            print(f"Error: {label} not found: {p}", file=sys.stderr)
            return 1

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
    k_neg = int(resolved["num_negatives_k"])
    hn = st["hard_negatives"]
    if hn["n_hard"] + hn["m_random"] != k_neg:
        print(
            f"Error: n_hard + m_random must equal num_negatives_k ({k_neg}).",
            file=sys.stderr,
        )
        return 1

    evaluate_legals_n = hn.get("evaluate_legals_n")
    if evaluate_legals_n is not None and evaluate_legals_n <= k_neg:
        print(
            f"Error: hard_negatives.evaluate_legals_n ({evaluate_legals_n}) must be > "
            f"num_negatives_k ({k_neg}) so the sampled set can include {k_neg} wrong moves.",
            file=sys.stderr,
        )
        return 1

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

    rep_tr: dict
    rep_va: dict
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

        print(f"Materializing train ({len(train_idx)} rows), use_hard_mining={use_hard}...", file=sys.stderr)
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
        print(f"Train report: {rep_tr}", file=sys.stderr)

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
        print(f"Val report: {rep_va}", file=sys.stderr)

        write_materialize_cache_manifest(
            manifest_path,
            cache_key=cache_key,
            key_sha256=key_sha256,
            train_h5_path=train_h5_path,
            val_h5_path=val_h5_path,
        )

    if rep_tr["n_written"] == 0:
        print("Error: no training rows materialized.", file=sys.stderr)
        return 1
    if rep_va["n_written"] == 0:
        print("Error: no val rows materialized.", file=sys.stderr)
        return 1

    assert_h5_k_matches(train_h5_path, k_neg)
    assert_h5_k_matches(val_h5_path, k_neg)

    model = load_jepa_from_checkpoint(prev_path, device=device)

    d = spec["defaults"]
    tr = st["train"]
    use_amp = bool(d.get("use_amp", True)) and device.type == "cuda"
    workers = int(d.get("dataloader_num_workers", 0))

    train_loader, val_loader = get_dataloaders(
        train_h5_path,
        val_h5_path,
        batch_size=int(tr["batch_size"]),
        num_workers=workers,
        in_memory=bool(d.get("in_memory", False)),
    )

    n_train, n_val = h5_transition_counts(train_h5_path, val_h5_path)
    train_meta = {
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
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Chess-JEPA staged training from move HDF5.")
    parser.add_argument("--model", type=str, required=True, help="Model name (jepa/model_configs/{name}.yaml)")
    parser.add_argument("--stage", type=int, required=True, help="0=init only; N>=1 trains stage N using stages[N-1]")
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu (default: auto)")
    parser.add_argument(
        "--rematerialize",
        action="store_true",
        help="Ignore materialized HDF5 cache and rebuild train/val JEPA files for this stage",
    )
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
    return cmd_train_stage(spec, args.stage, device, rematerialize=bool(args.rematerialize))


if __name__ == "__main__":
    raise SystemExit(main())
