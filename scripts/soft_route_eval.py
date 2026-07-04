#!/usr/bin/env python3
"""Compare HARD band-routing vs SOFT (elo-interpolated) routing for a MultiBandPolicy, top-1.

Each band head ~ represents its band center (b+50). Hard routing sends elo E to the head whose
band contains E (so 1500 -> the 1550-center head, a +50 over-shoot). Soft routing instead mixes the
two heads whose centers bracket E, weighted by proximity (1400 -> 50% 1350-center + 50% 1450-center),
by averaging their move *probabilities* then taking argmax. No retrain — inference only. Metric
matches history_ksweep (from top-1; to top-1 given true from; stored legal masks; history K=2).
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import h5py
from style_policy.multiband_policy import MultiBandPolicy
from style_policy.legal_mask import u64_to_mask

_NEG = float("-inf")


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val-h5", default="/mnt/eloquence_bulk/databases/wdl_validation_2025_05.h5")
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    dev = a.device

    ck = torch.load(a.ckpt, map_location=dev)
    model = MultiBandPolicy.from_config(ck["architecture"]); model.load_state_dict(ck["model"]); model.to(dev).eval()
    nb = model.n_bands
    b0 = int(model.bands[0])           # first band lower edge (1000)
    centers = torch.tensor([b + 50 for b in model.bands], dtype=torch.float32)  # 1050..2150

    with h5py.File(a.val_h5, "r") as f:
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

    band_of = model.head_index(elo).numpy()  # true containing-band, for per-band reporting
    # soft weights: klow = lower bracketing center index, whi = weight on klow+1
    ec = elo.clamp(int(centers[0]), int(centers[-1])).float()
    klow = torch.clamp(((ec - centers[0]) / 100).floor().long(), 0, nb - 2)
    whi = (ec - centers[klow]) / 100.0
    wlo = 1.0 - whi

    hard_mv = np.zeros(n, bool); soft_mv = np.zeros(n, bool)
    for i in range(0, n, a.batch):
        sl = slice(i, min(i + a.batch, n)); B = sl.stop - sl.start
        cls, squares = model.encode(packed[sl], hist=(hf[sl].to(dev), ht[sl].to(dev), hc[sl].to(dev)))
        fm = u64_to_mask(fmask_u[sl].to(dev)); tm = u64_to_mask(tmask_u[sl].to(dev))
        fs = from_sq[sl].to(dev); ts = to_sq[sl].to(dev)
        # all-head prob distributions (softmax over legal) for from and for to|true-from
        Pf = torch.empty(nb, B, 64, device=dev); Pt = torch.empty(nb, B, 64, device=dev)
        for g in range(nb):
            Pf[g] = torch.softmax(model.heads[g].from_logits(squares, cls).masked_fill(~fm, _NEG), dim=-1)
            Pt[g] = torch.softmax(model.heads[g].to_logits(squares, fs, cls).masked_fill(~tm, _NEG), dim=-1)
        rows = torch.arange(B)
        # HARD: gather the containing-band head
        r = torch.from_numpy(band_of[sl])
        hf_pred = Pf[r, rows].argmax(-1); ht_pred = Pt[r, rows].argmax(-1)
        hard_mv[sl] = ((hf_pred == fs) & (ht_pred == ts)).cpu().numpy()
        # SOFT: mix the two bracketing heads by weight
        kl = klow[sl]; wl = wlo[sl].unsqueeze(-1); wh = whi[sl].unsqueeze(-1)
        mf = wl * Pf[kl, rows] + wh * Pf[kl + 1, rows]
        mt = wl * Pt[kl, rows] + wh * Pt[kl + 1, rows]
        sf_pred = mf.argmax(-1); st_pred = mt.argmax(-1)
        soft_mv[sl] = ((sf_pred == fs) & (st_pred == ts)).cpu().numpy()

    print(f"ckpt={a.ckpt.split('/')[-1]}  n={n}")
    print(f"  HARD routing: move% = {100*hard_mv.mean():.2f}")
    print(f"  SOFT routing: move% = {100*soft_mv.mean():.2f}   (Δ {100*(soft_mv.mean()-hard_mv.mean()):+.2f})")
    print(f"  {'band':>9} {'hard':>6} {'soft':>6} {'Δ':>6}")
    for b in range(nb):
        m = band_of == b
        if m.any():
            h = 100*hard_mv[m].mean(); s = 100*soft_mv[m].mean()
            print(f"  {model.bands[b]:>4}-{model.bands[b]+99:<4} {h:>6.1f} {s:>6.1f} {s-h:>+6.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
