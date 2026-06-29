#!/usr/bin/env python3
"""Speed-only arch comparison: time full fwd+bwd/step for each (layers,d_model,dff,nhead) with the
deployed compile path. Synthetic batches; reports params, samp/s, VRAM. Quality NOT measured."""
from __future__ import annotations
import argparse, time
import numpy as np
import torch
from style_policy.model import BasePolicy
from style_policy.packed_codec import PACKED_BOARD_LEN


def _rand_packed(n, dev):
    p = np.zeros((n, PACKED_BOARD_LEN), dtype=np.uint8)
    p[:, :32] = np.random.randint(0, 13, size=(n, 32), dtype=np.uint8)
    p[:, 32] = np.random.randint(0, 32, size=n, dtype=np.uint8)
    p[:, 33] = 255
    return torch.from_numpy(p).to(dev)


def _time(fn, iters, dev):
    for _ in range(5):
        fn()
    torch.cuda.synchronize() if dev == "cuda" else None
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize() if dev == "cuda" else None
    return (time.perf_counter() - t0) * 1000 / iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--archs", nargs="+", default=["8:256:1024:8", "6:320:1280:8"],
                    help="each = layers:d_model:dff:nhead")
    a = ap.parse_args()
    dev = a.device
    print(f"device={dev} bs={a.bs} compile={'off' if a.no_compile else 'on'}")
    bs = a.bs
    packed = _rand_packed(bs, dev)
    fs = torch.randint(0, 64, (bs,), device=dev)
    flu = torch.full((bs,), -1, dtype=torch.int64, device=dev)
    tlu = torch.full((bs,), -1, dtype=torch.int64, device=dev)
    elo = torch.randint(10, 20, (bs,), device=dev)
    for spec in a.archs:
        nl, d, dff, nh = (int(x) for x in spec.split(":"))
        arch = {"d_model": d, "n_layers": nl, "nhead": nh, "dim_feedforward": dff,
                "dropout": 0.0, "head_hidden": 512, "elo_dim": 32, "n_elo_buckets": 40}
        model = BasePolicy.from_config(arch).to(dev); model.train()
        if not a.no_compile and dev == "cuda":
            model.encoder = torch.compile(model.encoder)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4, fused=(dev == "cuda"))
        nparams = sum(p.numel() for p in model.parameters())

        def step():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                fl, fm, tl, tm, vl = model.forward_policy(packed, fs, flu, tlu, elo_idx=elo)
                loss = fl.float().mean() + tl.float().mean() + vl.float().mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        torch.cuda.reset_peak_memory_stats() if dev == "cuda" else None
        ms = _time(step, a.iters, dev)
        mem = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0
        print(f"  {nl}x{d} dff{dff}: {nparams/1e6:5.1f}M params  {ms:6.1f} ms/step  "
              f"{bs/ms*1000:>7,.0f} samp/s  VRAM={mem:.2f}GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
