#!/usr/bin/env python3
"""Evaluate a band head's move-match across bands vs the shared head, on the same rows."""
import argparse, torch
from style_policy.band_head import BandHead, eval_band_head_row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="style_policy_checkpoints/base_64M/base_64M_stage_1.pt")
    ap.add_argument("--head", required=True)
    ap.add_argument("--val-h5", default="/mnt/eloquence_bulk/databases/wdl_validation_1M.h5")
    ap.add_argument("--bands", type=int, nargs="+", default=list(range(1000, 2000, 100)))
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    st = torch.load(a.head, map_location=a.device)
    head = BandHead(st["d_model"], st["hidden"]); head.load_state_dict(st["band_head"])
    rows = eval_band_head_row(a.checkpoint, head, a.val_h5, a.bands, device=a.device, n=a.n)
    trained_band = st["band"]
    print(f"band head trained @ {trained_band}  (spec = this head, shared = baseline @ each band)")
    print(f"{'band':>6} {'count':>6} {'spec%':>7} {'shared%':>8} {'edge':>6}")
    for b in a.bands:
        r = rows[b]
        print(f"{b:>6} {r['count']:>6} {r['spec']:>7.1f} {r['shared']:>8.1f} {r['spec']-r['shared']:>+6.1f}")

if __name__ == "__main__":
    raise SystemExit(main())
