#!/usr/bin/env python3
"""Train a MultiBandPolicy (shared encoder + per-band heads)."""
import argparse, torch
from style_policy.model_spec import load_spec
from style_policy.multiband_train import train_multiband

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="multiband_16M")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    rec = train_multiband(load_spec(a.model), a.device, resume=a.resume)
    print(rec)

if __name__ == "__main__":
    raise SystemExit(main())
