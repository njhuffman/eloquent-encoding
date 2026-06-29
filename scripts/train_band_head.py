#!/usr/bin/env python3
"""Train a band-specialized head on a frozen encoder."""
import argparse
from style_policy.band_head import train_band_head

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="style_policy_checkpoints/base_64M/base_64M_stage_1.pt")
    ap.add_argument("--band", type=int, required=True)
    ap.add_argument("--train-h5", default="/mnt/eloquence_bulk/databases/wdl_training_16M.h5")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--sample-n", type=int, default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    _, meta = train_band_head(a.checkpoint, a.band, a.train_h5, device=a.device, steps=a.steps,
                              batch_size=a.batch_size, sample_n=a.sample_n, lr=a.lr,
                              label_smoothing=a.label_smoothing, out=a.out)
    print("saved", a.out, meta)

if __name__ == "__main__":
    raise SystemExit(main())
