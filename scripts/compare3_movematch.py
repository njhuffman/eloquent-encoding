#!/usr/bin/env python3
"""3-way move-prediction (top-1, from AND to) per elo band, on the same val positions:
  (1) multiband band-routed head, (2) best conditioned model (base_64M @ band), (3) Maia2 @ band.
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import h5py
import chess
from style_policy.model import BasePolicy
from style_policy.band_head import BandHead
from style_policy.model_spec import elo_to_bucket
from style_policy.board_encode import packed_to_board, legal_from_u64, legal_to_u64
from style_policy.legal_mask import u64_to_mask
from style_policy.maia2_bot import load_maia2

_NEG = float("-inf")


def _mask(u64, dev):
    return u64_to_mask(torch.from_numpy(np.array([u64], dtype=np.uint64)).to(torch.int64)).to(dev)


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--multi-enc", default="style_policy_checkpoints/multiband_64M/multiband_64M_encoder.pt")
    ap.add_argument("--multi-head-dir", default="style_policy_checkpoints/multiband_64M/band_heads")
    ap.add_argument("--multi-prefix", default="multiband_64M_band_")
    ap.add_argument("--cond", default="style_policy_checkpoints/base_64M/base_64M_stage_1.pt")
    ap.add_argument("--val-h5", default="/mnt/eloquence_bulk/databases/wdl_validation_1M.h5")
    ap.add_argument("--bands", type=int, nargs="+", default=list(range(1000, 2000, 100)))
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device

    # multiband encoder + band heads
    mck = torch.load(a.multi_enc, map_location=dev)
    march = mck["architecture"]
    mmodel = BasePolicy.from_config(march); mmodel.load_state_dict(mck["model"], strict=False); mmodel.to(dev).eval()
    heads = {}
    for b in a.bands:
        st = torch.load(f"{a.multi_head_dir}/{a.multi_prefix}{b}.pt", map_location=dev)
        bh = BandHead(st["d_model"], st["hidden"]); bh.load_state_dict(st["band_head"]); heads[b] = bh.to(dev).eval()
    # conditioned model
    cck = torch.load(a.cond, map_location=dev)
    cmodel = BasePolicy.from_config(cck["architecture"]); cmodel.load_state_dict(cck["model"], strict=False); cmodel.to(dev).eval()
    n_elo = int(cck["architecture"]["n_elo_buckets"])
    bidx = {b: elo_to_bucket(torch.tensor([b]), n_elo).to(dev) for b in a.bands}
    # maia2
    maia, prep = load_maia2("rapid", device=("gpu" if str(dev).startswith("cuda") else "cpu"))
    from maia2 import inference

    with h5py.File(a.val_h5, "r") as f:
        n = min(a.n, f["packed_pre"].shape[0])
        packed = f["packed_pre"][:n]; hf = f["from_sq"][:n]; ht = f["to_sq"][:n]; elo = f["elo_to_move"][:n]

    counts = {b: 0 for b in a.bands}
    multi = {b: 0 for b in a.bands}; cond = {b: 0 for b in a.bands}; mm = {b: 0 for b in a.bands}

    def argmove(squares, ffn, tfn, board):
        fmask = _mask(legal_from_u64(board), dev)
        pf = int(ffn(squares).masked_fill(~fmask, _NEG).argmax())
        pt = int(tfn(squares, torch.tensor([pf], device=dev)).masked_fill(~_mask(legal_to_u64(board, pf), dev), _NEG).argmax())
        return pf, pt

    import time as _t; _t0 = _t.time()
    for i in range(n):
        if i and i % 1000 == 0:
            print(f"  ...{i}/{n}  ({i/(_t.time()-_t0):.0f} rows/s)", flush=True)
        board = packed_to_board(np.asarray(packed[i], np.uint8))
        if board.is_game_over():
            continue
        b = int(min(max(a.bands), max(min(a.bands), (int(elo[i]) // 100) * 100)))
        counts[b] += 1
        hfi, hti = int(hf[i]), int(ht[i])
        pk = torch.from_numpy(np.asarray(packed[i], np.uint8)[None]).to(dev)
        # multiband band-routed
        _, msq = mmodel.encode(pk)
        pf, pt = argmove(msq, lambda s: heads[b].from_logits(s), lambda s, x: heads[b].to_logits(s, x), board)
        if pf == hfi and pt == hti:
            multi[b] += 1
        # conditioned @ band
        _, csq = cmodel.encode(pk)
        pf, pt = argmove(csq, lambda s: cmodel.from_head(s, elo_idx=bidx[b]),
                         lambda s, x: cmodel.to_head(s, x, elo_idx=bidx[b]), board)
        if pf == hfi and pt == hti:
            cond[b] += 1
        # maia2 @ band (top-1 over legal)
        mp, _ = inference.inference_each(maia, prep, board.fen(), b, b)
        legal = {m.uci(): m for m in board.legal_moves}
        best = max(((u, p) for u, p in mp.items() if u in legal), key=lambda kv: kv[1], default=(None, 0))[0]
        if best is not None:
            mv = legal[best]
            if mv.from_square == hfi and mv.to_square == hti:
                mm[b] += 1

    def pct(d, b):
        return 100.0 * d[b] / counts[b] if counts[b] else 0.0
    print(f"\n3-way move-match top-1 (%, from AND to)  n={n}")
    print(f"{'band':>6} {'count':>6} {'multiband':>10} {'base_64M(cond)':>15} {'maia2':>8}")
    tot = {"multi": 0, "cond": 0, "mm": 0, "c": 0}
    for b in a.bands:
        print(f"{b:>6} {counts[b]:>6} {pct(multi,b):>10.1f} {pct(cond,b):>15.1f} {pct(mm,b):>8.1f}")
        tot["multi"] += multi[b]; tot["cond"] += cond[b]; tot["mm"] += mm[b]; tot["c"] += counts[b]
    c = max(tot["c"], 1)
    print(f"{'MEAN':>6} {tot['c']:>6} {100*tot['multi']/c:>10.1f} {100*tot['cond']/c:>15.1f} {100*tot['mm']/c:>8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
