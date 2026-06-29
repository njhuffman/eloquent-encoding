#!/usr/bin/env python3
"""Maia-diagonal check: does conditioning the policy at elo C predict band-C players' moves best?

For each validation position (binned by the mover's true 100-band), compute the model's greedy
top-1 move under every elo conditioning C and measure full-move match (from AND to) vs the human's
actual move. Prints an accuracy matrix [conditioning x true-band]; the diagonal should dominate each
column if the elo conditioning is calibrated.
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import h5py
from style_policy.model import BasePolicy
from style_policy.model_spec import elo_to_bucket
from style_policy.board_encode import packed_to_board, legal_from_u64, legal_to_u64
from style_policy.legal_mask import u64_to_mask

_NEG = float("-inf")


def _mask(u64, dev):
    return u64_to_mask(torch.from_numpy(np.array([u64], dtype=np.uint64)).to(torch.int64)).to(dev)


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="style_policy_checkpoints/base_64M/base_64M_stage_1.pt")
    ap.add_argument("--val-h5", default="/mnt/eloquence_bulk/databases/wdl_validation_1M.h5")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=10000)
    a = ap.parse_args()
    dev = a.device
    ck = torch.load(a.checkpoint, map_location=dev)
    m = BasePolicy.from_config(ck["architecture"]); m.load_state_dict(ck["model"], strict=False); m.to(dev).eval()
    n_elo = int(ck["architecture"]["n_elo_buckets"])
    bands = list(range(1000, 2000, 100))
    bidx = {C: elo_to_bucket(torch.tensor([C]), n_elo).to(dev) for C in bands}

    with h5py.File(a.val_h5, "r") as f:
        n = min(a.n, f["packed_pre"].shape[0])
        packed = f["packed_pre"][:n]; hf = f["from_sq"][:n]; ht = f["to_sq"][:n]; elo = f["elo_to_move"][:n]

    counts = {b: 0 for b in bands}
    match = {C: {b: 0 for b in bands} for C in bands}
    for i in range(n):
        board = packed_to_board(np.asarray(packed[i], np.uint8))
        if board.is_game_over():
            continue
        tb = int(min(1900, max(1000, (int(elo[i]) // 100) * 100)))
        counts[tb] += 1
        pk = torch.from_numpy(np.asarray(packed[i], np.uint8)[None]).to(dev)
        _, squares = m.encode(pk)
        fmask = _mask(legal_from_u64(board), dev)
        for C in bands:
            fl = m.from_head(squares, elo_idx=bidx[C])
            pf = int(fl.masked_fill(~fmask, _NEG).argmax())
            tl = m.to_head(squares, torch.tensor([pf], device=dev), elo_idx=bidx[C])
            pt = int(tl.masked_fill(~_mask(legal_to_u64(board, pf), dev), _NEG).argmax())
            if pf == int(hf[i]) and pt == int(ht[i]):
                match[C][tb] += 1

    print(f"move-match top-1 accuracy (%)  rows=conditioning, cols=true band   n={n}")
    print("cond \\ band  " + " ".join(f"{b:>5}" for b in bands))
    acc = {C: {b: (100.0 * match[C][b] / counts[b] if counts[b] else 0.0) for b in bands} for C in bands}
    colmax = {b: max(bands, key=lambda C: acc[C][b]) for b in bands}
    for C in bands:
        cells = []
        for b in bands:
            star = "*" if colmax[b] == C else " "
            cells.append(f"{acc[C][b]:4.1f}{star}")
        print(f"{C:>5}        " + " ".join(cells))
    # diagonal summary: per band, is the best conditioning the matching one? and the diagonal edge.
    print("\nper-band: best-conditioning (argmax), diagonal acc, mean off-diagonal, edge")
    diag_is_best = 0
    for b in bands:
        best = colmax[b]
        off = np.mean([acc[C][b] for C in bands if C != b])
        edge = acc[b][b] - off
        diag_is_best += int(best == b)
        print(f"  band {b}: best={best}  diag={acc[b][b]:.1f}  off-mean={off:.1f}  edge={edge:+.1f}")
    print(f"\ndiagonal is the single best conditioning in {diag_is_best}/{len(bands)} bands")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
