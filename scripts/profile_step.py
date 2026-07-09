"""Locate the training-step bottleneck: per-section timing (encoder / 12-band routed head-loss /
backward / opt) + end-to-end steps/s across batch sizes, eager vs compiled encoder.
Tells us whether the per-band head loop is worth batching, and the batch sweet spot."""
from __future__ import annotations
import argparse, time, collections, torch
from torch.utils.data import DataLoader
from style_policy.model_spec import load_spec, elo_to_bucket
from style_policy.multiband_policy import MultiBandPolicy
from style_policy.dataset import PackedMoveDataset
from style_policy.multiband_train import _routed_distill_loss
from style_policy.legal_mask import u64_to_mask
from style_policy.loss import wdl_ce

dev = "cuda"
def sync(): torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="multiband_ourdistill")
    ap.add_argument("--batches", default="256,512,1024")
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    spec = load_spec(a.model); arch = spec["architecture"]; n_elo = int(arch["n_elo_buckets"])
    batches = [int(x) for x in a.batches.split(",")]

    def get_batch(bs):
        ds = PackedMoveDataset(spec["train_h5"], sample_n=bs, seed=1)
        b = next(iter(DataLoader(ds, batch_size=bs, collate_fn=PackedMoveDataset.collate)))
        packed = b["packed_pre"].to(dev); elo = b["elo_to_move"]
        return dict(packed=packed, elo=elo, hidx=None,
                    fsq=b["from_sq"].to(dev), tsq=b["to_sq"].to(dev),
                    fmask=u64_to_mask(b["from_legal_u64"].to(dev)), tmask=u64_to_mask(b["to_legal_u64"].to(dev)),
                    mf=b["maia_from"].to(dev), mt=b["maia_to"].to(dev), result=b["result"].to(dev))

    def step(model, d, back=True):
        cls, sq = model.encode(d["packed"], hist=None)
        fl, tl = _routed_distill_loss(model, cls, sq, d["hidx"], d["fsq"], d["mf"], d["mt"], d["fmask"], d["tmask"])
        vl = wdl_ce(model.value_head(cls, elo_idx=elo_to_bucket(d["elo"], n_elo).to(dev)), d["result"])
        loss = fl + tl + vl
        if back: loss.backward()
        return loss

    # ---- section breakdown (eager), batch 512 ----
    model = MultiBandPolicy.from_config(arch).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    d = get_batch(256); d["hidx"] = model.head_index(d["elo"]).to(dev)
    for _ in range(3): step(model, d); opt.step(); opt.zero_grad()
    sync(); T = collections.defaultdict(float)
    for _ in range(a.iters):
        t0 = time.time(); cls, sq = model.encode(d["packed"], hist=None); sync(); t1 = time.time()
        fl, tl = _routed_distill_loss(model, cls, sq, d["hidx"], d["fsq"], d["mf"], d["mt"], d["fmask"], d["tmask"])
        vl = wdl_ce(model.value_head(cls, elo_idx=elo_to_bucket(d["elo"], n_elo).to(dev)), d["result"]); loss = fl + tl + vl
        sync(); t2 = time.time(); loss.backward(); sync(); t3 = time.time(); opt.step(); opt.zero_grad(); sync(); t4 = time.time()
        T["encoder(eager)"] += t1 - t0; T["heads+loss"] += t2 - t1; T["backward"] += t3 - t2; T["opt"] += t4 - t3
    print("=== EAGER section breakdown @ batch 256 (ms/step) ===")
    tot = 0
    for k in ["encoder(eager)", "heads+loss", "backward", "opt"]:
        ms = 1000 * T[k] / a.iters; tot += ms; print(f"  {k:16s}{ms:7.1f}")
    print(f"  {'TOTAL':16s}{tot:7.1f}  = {1000/tot:.2f} steps/s = {256*1000/tot:.0f} samples/s (eager b256)")

    del model, opt; torch.cuda.empty_cache()   # free eager model before the compiled sweep
    # ---- end-to-end steps/s, COMPILED encoder (matches training), across batch sizes ----
    print("\n=== COMPILED-encoder end-to-end (matches training) ===")
    for bs in batches:
        m = MultiBandPolicy.from_config(arch).to(dev).train()
        m.encoder = torch.compile(m.encoder)
        o = torch.optim.AdamW(m.parameters(), lr=3e-4)
        db = get_batch(bs); db["hidx"] = m.head_index(db["elo"]).to(dev)
        try:
            for _ in range(5): step(m, db); o.step(); o.zero_grad()   # compile warmup
            sync(); t0 = time.time()
            for _ in range(a.iters): step(m, db); o.step(); o.zero_grad()
            sync(); dt = (time.time() - t0) / a.iters
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"  batch {bs:5d}: {1000*dt:6.1f} ms/step = {1/dt:5.2f} steps/s = {bs/dt:6.0f} samples/s  (peak {mem:.1f} GB)")
        except RuntimeError as e:
            print(f"  batch {bs:5d}: OOM/err ({str(e)[:50]})")
        del m, o; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
