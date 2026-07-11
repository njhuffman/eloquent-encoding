"""Train a value head on the human world-model encoder to predict STOCKFISH's eval (sf_wdl, depth-8,
STM). Probe: can the encoder support a STRONG value function?
  --enc-lr 0  : FROZEN encoder (does the human-move representation already carry strong-eval signal?)
  --enc-lr >0 : SUPERVISED fine-tune the encoder too (stable, unlike RL) — does adapting the features
                break the r~0.86 frozen cap and yield a strong value bot?
Rate the result with rate_value_bot.py --sf-value-head (and --ckpt the fine-tuned model)."""
from __future__ import annotations
import argparse, sys, os, numpy as np, torch, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_policy.multiband_policy import MultiBandPolicy
from style_policy.value_head import WDLHead

DEV = "cuda"


def _corr_and_ce(model, head, packed, Y, idx):
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pv, ce, n = [], 0.0, 0
        for i in range(0, len(idx), 1024):
            b = idx[i:i+1024]
            pk = torch.from_numpy(packed[b].astype(np.int64)).to(DEV)
            cls, _ = model.encode(pk, hist=None)
            logp = torch.log_softmax(head(cls.float()), -1)
            ce += -(Y[b].to(DEV) * logp).sum().item(); n += len(b)
            p = logp.exp(); pv.append((p[:, 2] - p[:, 0]).cpu())
    pred = torch.cat(pv).numpy(); sf = (Y[idx][:, 2] - Y[idx][:, 0]).numpy()
    return ce / (n * 1.0), float(np.corrcoef(pred, sf)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="style_policy_checkpoints/multiband_ourdistill/multiband_ourdistill.pt")
    ap.add_argument("--base", default="/mnt/eloquence_bulk/databases/wdl_validation_2025_05.h5")
    ap.add_argument("--sidecar", default="/mnt/eloquence_bulk/databases/wdl_validation_2025_05.sf_eval.h5")
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--enc-lr", type=float, default=0.0)
    ap.add_argument("--out", default="style_policy_checkpoints/sf_value_head.pt")
    ap.add_argument("--model-out", default="")     # save fine-tuned/scratch full model (for the value bot)
    ap.add_argument("--from-scratch", action="store_true")   # RANDOM init encoder (no human warm-start)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location=DEV); arch = ck["architecture"]
    model = MultiBandPolicy.from_config(arch)
    if not a.from_scratch:
        model.load_state_dict(ck["model"], strict=False)     # human warm-start (skip for from-scratch)
    model.to(DEV).eval()
    for p in model.parameters(): p.requires_grad_(False)
    ft = a.enc_lr > 0 or a.from_scratch
    if ft:
        for p in model.encoder.parameters(): p.requires_grad_(True)
        model.encoder.train()
    fb = h5py.File(a.base, "r"); fs = h5py.File(a.sidecar, "r")
    rows = fs["row_index"][:]; packed = fb["packed_pre"][:][rows]
    wdl = fs["sf_wdl"][:].astype(np.float32) / 1000.0
    keep = wdl.sum(1) > 0.5; packed, wdl = packed[keep], torch.from_numpy(wdl[keep])
    print(f"{len(packed):,} Stockfish-labeled positions | encoder {'FINE-TUNE' if ft else 'FROZEN'}", flush=True)
    d, h = int(arch["d_model"]), int(arch["head_hidden"])
    head = WDLHead(d_model=d, hidden=h, elo_dim=0).to(DEV)
    n = len(packed); p = torch.randperm(n); ntr = int(0.9 * n); tri, tei = p[:ntr].numpy(), p[ntr:].numpy()
    Y = wdl
    groups = [{"params": head.parameters(), "lr": 1e-3}]
    if ft: groups.append({"params": model.encoder.parameters(), "lr": a.enc_lr})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    tp = list(head.parameters()) + (list(model.encoder.parameters()) if ft else [])
    for ep in range(a.epochs):
        head.train(); order = np.random.permutation(tri)
        for i in range(0, len(order), 512):
            b = order[i:i+512]
            ctx = torch.enable_grad() if ft else torch.no_grad()
            with ctx, torch.amp.autocast("cuda", dtype=torch.bfloat16):
                cls, _ = model.encode(torch.from_numpy(packed[b].astype(np.int64)).to(DEV), hist=None)
            loss = -(Y[b].to(DEV) * torch.log_softmax(head(cls.float()), -1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(tp, 1.0); opt.step()
    head.eval(); model.eval()
    ce, r = _corr_and_ce(model, head, packed, Y, tei)
    print(f"held-out soft-CE {ce:.3f} | value corr(pred, stockfish) r={r:.3f}")
    torch.save({"value_head": head.state_dict(), "d_model": d, "hidden": h}, a.out)
    if ft and a.model_out:
        torch.save({"architecture": arch, "model": model.state_dict()}, a.model_out)
        print(f"saved fine-tuned model -> {a.model_out}")
    print(f"saved value head -> {a.out}")


if __name__ == "__main__":
    main()
