#!/usr/bin/env python3
"""Decompose a training step: unpack (cpu round-trip) vs encoder fwd+bwd, and test quick wins
(torch.compile, bigger batch, unpack-on-gpu). Synthetic packed batches; no HDF5 needed."""
from __future__ import annotations
import argparse, time
import numpy as np
import torch
from style_policy.model import BasePolicy
from style_policy.model_spec import load_spec
from style_policy.packed_codec import packed_to_board_tensor, PACKED_BOARD_LEN


def _rand_packed(n, dev):
    p = np.zeros((n, PACKED_BOARD_LEN), dtype=np.uint8)
    p[:, :32] = np.random.randint(0, 13, size=(n, 32), dtype=np.uint8)  # nibbles 0..12 (low only, valid)
    p[:, 32] = np.random.randint(0, 32, size=n, dtype=np.uint8)
    p[:, 33] = 255
    return torch.from_numpy(p).to(dev)


def _time(fn, iters, dev):
    for _ in range(5):
        fn()
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if dev == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wdl_16M")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[256, 512, 1024])
    a = ap.parse_args()
    spec = load_spec(a.model); arch = spec["architecture"]; dev = a.device
    model = BasePolicy.from_config(arch).to(dev); model.train()
    n_elo = int(arch["n_elo_buckets"])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

    def full_step(packed, fs, ts, flu, tlu, res, elo):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            fl, fm, tl, tm, vl = model.forward_policy(packed, fs, flu, tlu, elo_idx=elo)
            loss = (fl.float().mean() + tl.float().mean() + vl.float().mean())
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    print(f"model={a.model} params={sum(p.numel() for p in model.parameters())/1e6:.1f}M device={dev}")
    for bs in a.batch_sizes:
        packed = _rand_packed(bs, dev)
        fs = torch.randint(0, 64, (bs,), device=dev)
        ts = torch.randint(0, 64, (bs,), device=dev)
        flu = torch.full((bs,), -1, dtype=torch.int64, device=dev)  # all-legal mask
        tlu = torch.full((bs,), -1, dtype=torch.int64, device=dev)
        res = torch.randint(0, 3, (bs,), device=dev)
        elo = torch.randint(10, 20, (bs,), device=dev)
        try:
            t_unpack = _time(lambda: packed_to_board_tensor(packed).to(dev), a.iters, dev)
            t_full = _time(lambda: full_step(packed, fs, ts, flu, tlu, res, elo), a.iters, dev)
            mem = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0
            torch.cuda.reset_peak_memory_stats() if dev == "cuda" else None
            print(f"  bs={bs:>4}: full={t_full:6.1f}ms  unpack={t_unpack:5.1f}ms ({100*t_unpack/t_full:.0f}% of step)"
                  f"  -> {bs/t_full*1000:>7,.0f} samp/s  VRAM={mem:.2f}GB")
        except RuntimeError as e:
            print(f"  bs={bs:>4}: OOM/err: {str(e)[:60]}")
            break

    # torch.compile at bs256
    print("\n[torch.compile test @ bs256]")
    bs = 256
    packed = _rand_packed(bs, dev)
    fs = torch.randint(0, 64, (bs,), device=dev); ts = torch.randint(0, 64, (bs,), device=dev)
    flu = torch.full((bs,), -1, dtype=torch.int64, device=dev); tlu = torch.full((bs,), -1, dtype=torch.int64, device=dev)
    res = torch.randint(0, 3, (bs,), device=dev); elo = torch.randint(10, 20, (bs,), device=dev)
    base = _time(lambda: full_step(packed, fs, ts, flu, tlu, res, elo), a.iters, dev)
    try:
        model.encoder = torch.compile(model.encoder)
        comp = _time(lambda: full_step(packed, fs, ts, flu, tlu, res, elo), a.iters, dev)
        print(f"  eager={base:.1f}ms  compiled(encoder)={comp:.1f}ms  speedup={base/comp:.2f}x")
    except Exception as e:
        print(f"  compile failed: {str(e)[:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
