#!/usr/bin/env python3
"""Top-1 move-match for a CONDITIONED BasePolicy checkpoint, using the SAME metric as
history_ksweep.py (from = argmax over legal; to = argmax over legal given the TRUE from;
move = both correct; stored legal masks). Lets a conditioned model be compared apples-to-apples
against a multiband model's K-sweep numbers on the same val positions.
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import h5py
from style_policy.model import BasePolicy
from style_policy.model_spec import elo_to_bucket
from style_policy.legal_mask import u64_to_mask

_NEG = float("-inf")


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val-h5", default="/mnt/eloquence_bulk/databases/wdl_validation_2025_05.h5")
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = a.device

    ck = torch.load(a.ckpt, map_location=dev)
    model = BasePolicy.from_config(ck["architecture"])
    model.load_state_dict(ck["model"], strict=False)
    model.to(dev).eval()
    n_elo = int(ck["architecture"]["n_elo_buckets"])

    with h5py.File(a.val_h5, "r") as f:
        n = min(a.n, int(f["packed_pre"].shape[0]))
        packed = torch.from_numpy(f["packed_pre"][:n].astype(np.uint8))
        from_sq = torch.from_numpy(f["from_sq"][:n].astype(np.int64))
        to_sq = torch.from_numpy(f["to_sq"][:n].astype(np.int64))
        fmask_u = torch.from_numpy(np.array(f["from_legal_u64"][:n], dtype=np.uint64)).to(torch.int64)
        tmask_u = torch.from_numpy(np.array(f["to_legal_u64"][:n], dtype=np.uint64)).to(torch.int64)
        elo = torch.from_numpy(f["elo_to_move"][:n].astype(np.int64))
        hc0 = torch.from_numpy(f["hist_cap"][:n, 0].astype(np.int64)) if "hist_cap" in f else torch.zeros(n, dtype=torch.int64)

    reactive = (hc0 > 0).numpy()
    band_of = ((elo.clamp(1000, 2199) - 1000) // 100).numpy()  # 0..11, matches history_ksweep bands
    mv = np.zeros(n, bool); fr = np.zeros(n, bool); to = np.zeros(n, bool)
    for i in range(0, n, a.batch):
        sl = slice(i, min(i + a.batch, n))
        cls, squares = model.encode(packed[sl].to(dev))
        eidx = elo_to_bucket(elo[sl], n_elo).to(dev)
        fs = from_sq[sl].to(dev); ts = to_sq[sl].to(dev)
        fm = u64_to_mask(fmask_u[sl].to(dev)); tm = u64_to_mask(tmask_u[sl].to(dev))
        fpred = model.from_head(squares, elo_idx=eidx).masked_fill(~fm, _NEG).argmax(-1)
        tpred = model.to_head(squares, fs, elo_idx=eidx).masked_fill(~tm, _NEG).argmax(-1)
        fb = (fpred == fs).cpu().numpy(); tb = (tpred == ts).cpu().numpy()
        fr[sl] = fb; to[sl] = tb; mv[sl] = fb & tb

    print(f"ckpt={a.ckpt.split('/')[-1]}  val={a.val_h5.split('/')[-1]}  n={n}")
    print(f"move%={100*mv.mean():.2f}  from%={100*fr.mean():.2f}  to%={100*to.mean():.2f}  "
          f"reactive move%={100*mv[reactive].mean():.2f}" if reactive.any() else "")
    lo = mv[band_of <= 9]  # bands 1000-1999 only (base models were trained here)
    print(f"  1000-1999 only: move%={100*lo.mean():.2f}  (n={lo.size})")
    per = " ".join(f"{100*mv[band_of == b].mean():4.1f}" if (band_of == b).any() else "  - " for b in range(12))
    print(f"  per-band move% (1000..2100): {per}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
