#!/usr/bin/env python3
"""Move-match diagonal for FROZEN per-band heads vs the conditioned head, on the same rows.

For each validation position (binned by the mover's true band), compute the greedy top-1 move
under (a) each frozen per-band head and (b) the shared elo-conditioned head at each band. Builds
two matrices [predictor x true-band] and reports diagonal sharpness (column-best count + mean edge)
for each, so we can see whether the dedicated heads form a sharper move-match diagonal than the
elo embedding. Encodes once per position (the cost), applies all predictors. CPU-friendly.
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import h5py
from style_policy.model import BasePolicy
from style_policy.band_head import BandHead
from style_policy.model_spec import elo_to_bucket
from style_policy.board_encode import packed_to_board, legal_from_u64, legal_to_u64
from style_policy.legal_mask import u64_to_mask

_NEG = float("-inf")


def _mask(u64, dev):
    return u64_to_mask(torch.from_numpy(np.array([u64], dtype=np.uint64)).to(torch.int64)).to(dev)


def _sharpness(acc, bands, label):
    """acc[predictor][band] -> print matrix + per-band best/edge + diagonal-best count."""
    colbest = {b: max(bands, key=lambda p: acc[p][b]) for b in bands}
    print(f"\n=== {label}: move-match top-1 (%)  rows=predictor, cols=true band ===")
    print("pred \\ band " + " ".join(f"{b:>5}" for b in bands))
    for p in bands:
        cells = " ".join(f"{acc[p][b]:4.1f}{'*' if colbest[b]==p else ' '}" for b in bands)
        print(f"{p:>5}       {cells}")
    diag_best = sum(int(colbest[b] == b) for b in bands)
    edges = [acc[b][b] - np.mean([acc[p][b] for p in bands if p != b]) for b in bands]
    print(f"  diagonal is column-best in {diag_best}/{len(bands)} bands; mean diag edge = {np.mean(edges):+.2f}%")
    return diag_best, float(np.mean(edges))


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="style_policy_checkpoints/base_64M/base_64M_stage_1.pt")
    ap.add_argument("--head-dir", default="style_policy_checkpoints/band_heads")
    ap.add_argument("--prefix", default="base_64M_band_")
    ap.add_argument("--bands", type=int, nargs="+", default=list(range(1000, 2000, 100)))
    ap.add_argument("--val-h5", default="/mnt/eloquence_bulk/databases/wdl_validation_1M.h5")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    dev = a.device
    ck = torch.load(a.checkpoint, map_location=dev)
    arch = ck["architecture"]; n_elo = int(arch["n_elo_buckets"])
    model = BasePolicy.from_config(arch); model.load_state_dict(ck["model"], strict=False)
    model.to(dev).eval()
    d, h = int(arch["d_model"]), int(arch["head_hidden"])
    heads = {}
    for b in a.bands:
        st = torch.load(f"{a.head_dir}/{a.prefix}{b}.pt", map_location=dev)
        bh = BandHead(st["d_model"], st["hidden"]); bh.load_state_dict(st["band_head"])
        heads[b] = bh.to(dev).eval()
    bidx = {c: elo_to_bucket(torch.tensor([c]), n_elo).to(dev) for c in a.bands}

    with h5py.File(a.val_h5, "r") as f:
        n = min(a.n, f["packed_pre"].shape[0])
        packed = f["packed_pre"][:n]; hf = f["from_sq"][:n]; ht = f["to_sq"][:n]; elo = f["elo_to_move"][:n]

    counts = {b: 0 for b in a.bands}
    frozen = {p: {b: 0 for b in a.bands} for p in a.bands}
    cond = {p: {b: 0 for b in a.bands} for p in a.bands}
    for i in range(n):
        board = packed_to_board(np.asarray(packed[i], np.uint8))
        if board.is_game_over():
            continue
        tb = int(min(max(a.bands), max(min(a.bands), (int(elo[i]) // 100) * 100)))
        counts[tb] += 1
        pk = torch.from_numpy(np.asarray(packed[i], np.uint8)[None]).to(dev)
        _, squares = model.encode(pk)
        fmask = _mask(legal_from_u64(board), dev)
        hfi, hti = int(hf[i]), int(ht[i])
        for p in a.bands:
            # frozen head p
            pf = int(heads[p].from_logits(squares).masked_fill(~fmask, _NEG).argmax())
            pt = int(heads[p].to_logits(squares, torch.tensor([pf], device=dev))
                     .masked_fill(~_mask(legal_to_u64(board, pf), dev), _NEG).argmax())
            if pf == hfi and pt == hti:
                frozen[p][tb] += 1
            # conditioned head @ p
            cf = int(model.from_head(squares, elo_idx=bidx[p]).masked_fill(~fmask, _NEG).argmax())
            ct = int(model.to_head(squares, torch.tensor([cf], device=dev), elo_idx=bidx[p])
                     .masked_fill(~_mask(legal_to_u64(board, cf), dev), _NEG).argmax())
            if cf == hfi and ct == hti:
                cond[p][tb] += 1

    accf = {p: {b: (100.0 * frozen[p][b] / counts[b] if counts[b] else 0.0) for b in a.bands} for p in a.bands}
    accc = {p: {b: (100.0 * cond[p][b] / counts[b] if counts[b] else 0.0) for b in a.bands} for p in a.bands}
    print(f"n={n}  per-band counts: {counts}")
    fb, fe = _sharpness(accf, a.bands, "FROZEN per-band heads")
    cb, ce = _sharpness(accc, a.bands, "CONDITIONED head (elo embedding)")
    print(f"\nSUMMARY  frozen: diag-best {fb}/{len(a.bands)}, mean edge {fe:+.2f}%   |   "
          f"conditioned: diag-best {cb}/{len(a.bands)}, mean edge {ce:+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
