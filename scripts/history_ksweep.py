#!/usr/bin/env python3
"""Inference-time history K-sweep for a trained MultiBandPolicy with last-move history.

One trained model (trained with graded history-horizon dropout), swept at inference over
K = 0..n_history_ply most-recent plies fed (the rest marked absent) → the diminishing-returns
curve, with NO retraining. Reports top-1 move-match (from & teacher-forced to, matching the
training loss) overall, per band, and on a REACTIVE subset (ply-1 was a capture — where history
should help most, e.g. recaptures). Use it to pick n_history_ply for the full run.

Usage:
  python scripts/history_ksweep.py --ckpt style_policy_checkpoints/multiband_history_16M/multiband_history_16M.pt
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import h5py
from style_policy.multiband_policy import MultiBandPolicy
from style_policy.legal_mask import u64_to_mask

_NEG = float("-inf")


def _truncate(hf, ht, hc, k):
    """Keep the k most-recent plies (newest-first); mark the rest absent (-1,-1,0)."""
    hf, ht, hc = hf.clone(), ht.clone(), hc.clone()
    if k < hf.shape[1]:
        hf[:, k:] = -1
        ht[:, k:] = -1
        hc[:, k:] = 0
    return hf, ht, hc


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="joint MultiBandPolicy checkpoint ({name}.pt)")
    ap.add_argument("--val-h5", default="/mnt/eloquence_bulk/databases/wdl_validation_2025_05.h5")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = a.device

    ck = torch.load(a.ckpt, map_location=dev)
    model = MultiBandPolicy.from_config(ck["architecture"])
    model.load_state_dict(ck["model"])
    model.to(dev).eval()
    n_ply = int(ck["architecture"].get("n_history_ply", 4))
    bands = list(model.bands)

    with h5py.File(a.val_h5, "r") as f:
        if "hist_from" not in f:
            raise SystemExit("val set has no history columns — regenerate val WITH history.")
        n = min(a.n, int(f["packed_pre"].shape[0]))
        packed = torch.from_numpy(f["packed_pre"][:n].astype(np.uint8))
        from_sq = torch.from_numpy(f["from_sq"][:n].astype(np.int64))
        to_sq = torch.from_numpy(f["to_sq"][:n].astype(np.int64))
        fmask_u = torch.from_numpy(np.array(f["from_legal_u64"][:n], dtype=np.uint64)).to(torch.int64)
        tmask_u = torch.from_numpy(np.array(f["to_legal_u64"][:n], dtype=np.uint64)).to(torch.int64)
        elo = torch.from_numpy(f["elo_to_move"][:n].astype(np.int64))
        hf = torch.from_numpy(f["hist_from"][:n].astype(np.int64))
        ht = torch.from_numpy(f["hist_to"][:n].astype(np.int64))
        hc = torch.from_numpy(f["hist_cap"][:n].astype(np.int64))

    reactive = (hc[:, 0] > 0).numpy()  # ply-1 was a capture
    band_idx = model.head_index(elo).numpy()
    print(f"val={a.val_h5.split('/')[-1]}  n={n}  n_history_ply={n_ply}  "
          f"reactive(ply-1 capture)={reactive.sum()} ({100*reactive.mean():.1f}%)")
    print(f"{'K':>2} | {'move%':>6} {'from%':>6} {'to%':>6} | {'reactive move%':>14} | per-band move% (1000..)")

    for k in range(0, n_ply + 1):
        hfk, htk, hck = _truncate(hf, ht, hc, k)
        mv = np.zeros(n, bool); fr = np.zeros(n, bool); tob = np.zeros(n, bool)
        for i in range(0, n, a.batch):
            sl = slice(i, min(i + a.batch, n))
            hidx = model.head_index(elo[sl]).to(dev)
            cls, squares = model.encode(packed[sl], hist=(hfk[sl].to(dev), htk[sl].to(dev), hck[sl].to(dev)))
            fm = u64_to_mask(fmask_u[sl].to(dev)); tm = u64_to_mask(tmask_u[sl].to(dev))
            fs = from_sq[sl].to(dev); ts = to_sq[sl].to(dev)
            fpred = torch.zeros(fs.shape[0], dtype=torch.long, device=dev)
            tpred = torch.zeros(fs.shape[0], dtype=torch.long, device=dev)
            for g in range(model.n_bands):
                m = hidx == g
                if not bool(m.any()):
                    continue
                sq = squares[m]; cl = cls[m]
                fl = model.heads[g].from_logits(sq, cl).masked_fill(~fm[m], _NEG)
                fpred[m] = fl.argmax(-1)
                tl = model.heads[g].to_logits(sq, fs[m], cl).masked_fill(~tm[m], _NEG)
                tpred[m] = tl.argmax(-1)
            fb = (fpred == fs).cpu().numpy(); tb = (tpred == ts).cpu().numpy()
            fr[sl] = fb; tob[sl] = tb; mv[sl] = fb & tb
        per_band = " ".join(
            f"{100*mv[band_idx == b].mean():4.1f}" if (band_idx == b).any() else "  - "
            for b in range(len(bands))
        )
        rm = f"{100*mv[reactive].mean():13.2f}%" if reactive.any() else "         n/a "
        print(f"{k:>2} | {100*mv.mean():5.2f}% {100*fr.mean():5.2f}% {100*tob.mean():5.2f}% | {rm} | {per_band}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
