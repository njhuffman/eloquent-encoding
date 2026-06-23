#!/usr/bin/env python3
"""Evaluate the WDL head: log-loss vs the per-elo-bucket prior, WDL accuracy, and policy
full-move top-1 (the joint policy metric) on a packed-with-result val set."""
from __future__ import annotations
import argparse, math
import numpy as np
import torch
from torch.utils.data import DataLoader
from style_policy.model import BasePolicy
from style_policy.dataset import PackedMoveDataset
from style_policy.model_spec import elo_to_bucket
from style_policy.loss import joint_top1, wdl_accuracy


def prior_logloss_from_results(results: np.ndarray, buckets: np.ndarray) -> float:
    """Log-loss of predicting each row's per-elo-bucket marginal W/D/L distribution."""
    results = np.asarray(results, dtype=np.int64)
    buckets = np.asarray(buckets, dtype=np.int64)
    eps = 1e-12
    total = 0.0
    for b in np.unique(buckets):
        mask = buckets == b
        counts = np.bincount(results[mask], minlength=3).astype(np.float64)
        p = counts / counts.sum()
        total += float(-np.sum(np.log(np.maximum(p[results[mask]], eps))))
    return total / len(results)


@torch.no_grad()
def evaluate(checkpoint_path: str, val_h5: str, *, device: str = "cpu", sample_n=None) -> dict:
    ck = torch.load(checkpoint_path, map_location=device)
    model = BasePolicy.from_config(ck["architecture"]); model.load_state_dict(ck["model"]); model.to(device).eval()
    n_elo = int(ck["architecture"]["n_elo_buckets"])
    ds = PackedMoveDataset(val_h5, sample_n=sample_n)
    dl = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=PackedMoveDataset.collate)
    tot_ll = tot_acc = tot_top1 = 0.0
    nb = 0
    all_results = []
    all_buckets = []
    for batch in dl:
        elo_idx = elo_to_bucket(batch["elo_to_move"], n_elo).to(device)
        fl, fm, tl, tm, vlog = model.forward_policy(
            batch["packed_pre"].to(device), batch["from_sq"].to(device),
            batch["from_legal_u64"].to(device), batch["to_legal_u64"].to(device), elo_idx=elo_idx)
        result = batch["result"].to(device)
        tot_ll += torch.nn.functional.cross_entropy(vlog, result.long()).item()
        tot_acc += wdl_accuracy(vlog, result)
        tot_top1 += joint_top1(fl, batch["from_sq"].to(device), fm, tl, batch["to_sq"].to(device), tm)
        all_results.append(batch["result"].cpu().numpy())
        all_buckets.append(elo_idx.cpu().numpy())
        nb += 1
    nb = max(nb, 1)
    results = np.concatenate(all_results)
    buckets = np.concatenate(all_buckets)
    return {"wdl_logloss": tot_ll / nb, "wdl_acc": tot_acc / nb,
            "prior_logloss": prior_logloss_from_results(results, buckets),
            "full_top1": tot_top1 / nb, "n": int(len(results))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-h5", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sample-n", type=int, default=200000)
    args = ap.parse_args()
    print(evaluate(args.checkpoint, args.val_h5, device=args.device, sample_n=args.sample_n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
