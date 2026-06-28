#!/usr/bin/env python3
"""End-to-end training throughput probe for style_policy: decomposes IO vs compute so we can
size a single-epoch dataset (samples/sec x wall-clock = unique samples) and find the bottleneck
before a long run. Times (a) data-only iteration and (b) full fwd+bwd+opt over the REAL loader.
"""
from __future__ import annotations
import argparse, time
import torch
from torch.utils.data import DataLoader
from style_policy.model import BasePolicy
from style_policy.dataset import PackedMoveDataset
from style_policy.model_spec import load_spec
from style_policy.training_loop import _step_loss


def _loader(h5, bs, workers, *, pin, persistent, prefetch, sample_n, seed=1):
    ds = PackedMoveDataset(h5, sample_n=sample_n, seed=seed)
    kw = {}
    if workers > 0:
        kw = {"persistent_workers": persistent, "prefetch_factor": prefetch}
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=workers,
                    collate_fn=PackedMoveDataset.collate, pin_memory=pin, **kw)
    return dl


def _count(dl, n_batches):
    t0 = time.perf_counter(); seen = 0
    for i, b in enumerate(dl):
        seen += b["from_sq"].shape[0]
        if i + 1 >= n_batches:
            break
    return seen, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wdl_16M", help="config name under model_configs/")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batches", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=0, help="override config batch_size")
    ap.add_argument("--workers", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--sample-n", type=int, default=400000, help="subsample rows (keeps index build fast)")
    a = ap.parse_args()
    spec = load_spec(a.model)
    arch = spec["architecture"]; n_elo = int(arch["n_elo_buckets"])
    stage = spec["stages"][0]
    bs = a.batch_size or int(stage["batch_size"])
    h5 = spec["train_h5"]; dev = a.device
    print(f"model={a.model} batch_size={bs} h5={h5} device={dev}")
    print(f"arch: d_model={arch['d_model']} n_layers={arch['n_layers']} dff={arch['dim_feedforward']}")

    # ---- data-only, sweep workers (and pin/persistent on the best) ----
    print("\n[data-only iteration]  samples/sec by num_workers")
    best_w, best_rate = a.workers[0], 0.0
    for w in a.workers:
        dl = _loader(h5, bs, w, pin=True, persistent=False, prefetch=4, sample_n=a.sample_n)
        _count(dl, 5)  # warm workers
        seen, dt = _count(dl, a.batches)
        rate = seen / dt
        print(f"  workers={w:>2}: {rate:>9,.0f} samp/s  ({dt*1000/a.batches:.1f} ms/batch)")
        if rate > best_rate:
            best_rate, best_w = rate, w

    # ---- full training step at best worker count ----
    model = BasePolicy.from_config(arch).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"\nmodel params: {nparams/1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    dl = _loader(h5, bs, best_w, pin=True, persistent=True, prefetch=6, sample_n=a.sample_n)
    model.train()
    it = iter(dl)

    def step(batch):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            loss, _ = _step_loss(model, batch, dev, n_elo, 0.0, 1.0)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    for _ in range(5):  # warmup
        step(next(it))
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter(); seen = 0
    for _ in range(a.batches):
        b = next(it); seen += b["from_sq"].shape[0]; step(b)
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    full_rate = seen / dt
    print(f"\n[full step, workers={best_w}]  {full_rate:,.0f} samp/s  ({dt*1000/a.batches:.1f} ms/batch)")
    if dev == "cuda":
        print(f"  peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB / 4.0 GB")
    print(f"\n=> data-only best: {best_rate:,.0f} samp/s (workers={best_w}); full step: {full_rate:,.0f} samp/s")
    verdict = "IO-BOUND (data < compute; fix loader)" if best_rate < full_rate * 1.3 else "COMPUTE-BOUND (GPU saturated)"
    print(f"=> {verdict}")
    for hrs in (1, 6, 24):
        print(f"   at {full_rate:,.0f} samp/s: {full_rate*3600*hrs/1e6:,.0f}M samples in {hrs}h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
