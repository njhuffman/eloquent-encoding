"""Train a fresh value head on the FROZEN human world-model encoder to predict STOCKFISH's eval
(sf_wdl, depth-8, STM). Probe: can the frozen encoder's features support a STRONG value function?
If the resulting 1-ply value bot beats the human-value bot (~1869), the human targets were the
limiter, not the encoder. Uses the existing Stockfish sidecar."""
from __future__ import annotations
import argparse, sys, os, numpy as np, torch, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_policy.multiband_policy import MultiBandPolicy
from style_policy.value_head import WDLHead

DEV = "cuda"


@torch.no_grad()
def encode_cls(model, packed, bs=1024):
    out = []
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(packed), bs):
            pk = torch.from_numpy(packed[i:i+bs].astype(np.int64)).to(DEV)
            c, _ = model.encode(pk, hist=None); out.append(c.float().cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="style_policy_checkpoints/multiband_ourdistill/multiband_ourdistill.pt")
    ap.add_argument("--base", default="/mnt/eloquence_bulk/databases/wdl_validation_1M.h5")
    ap.add_argument("--sidecar", default="/mnt/eloquence_bulk/databases/wdl_validation_1M.sf_eval.h5")
    ap.add_argument("--epochs", type=int, default=200); ap.add_argument("--out", default="style_policy_checkpoints/sf_value_head.pt")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location=DEV); arch = ck["architecture"]
    model = MultiBandPolicy.from_config(arch); model.load_state_dict(ck["model"], strict=False)
    model.to(DEV).eval()
    for p in model.parameters(): p.requires_grad_(False)
    fb = h5py.File(a.base, "r"); fs = h5py.File(a.sidecar, "r")
    rows = fs["row_index"][:]
    packed = fb["packed_pre"][:][rows]
    wdl = fs["sf_wdl"][:].astype(np.float32) / 1000.0                      # (N,3) loss/draw/win prob
    keep = wdl.sum(1) > 0.5                                                # drop terminal (all-zero)
    packed, wdl = packed[keep], torch.from_numpy(wdl[keep])
    print(f"{len(packed):,} Stockfish-labeled positions; encoding (frozen) ...", flush=True)
    cls = encode_cls(model, packed)
    d, h = int(arch["d_model"]), int(arch["head_hidden"])
    head = WDLHead(d_model=d, hidden=h, elo_dim=0).to(DEV)
    n = len(cls); p = torch.randperm(n); ntr = int(0.85 * n); tri, tei = p[:ntr], p[ntr:]
    X, Y = cls.to(DEV), wdl.to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(a.epochs):
        head.train(); pp = tri[torch.randperm(len(tri))]
        for i in range(0, len(tri), 1024):
            b = pp[i:i+1024]
            loss = -(Y[b] * torch.log_softmax(head(X[b]), -1)).sum(1).mean()  # soft cross-entropy
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    with torch.no_grad():
        vl = -(Y[tei] * torch.log_softmax(head(X[tei]), -1)).sum(1).mean().item()
        # correlation of predicted value (P(win)-P(loss)) vs Stockfish value on held-out
        pv = torch.softmax(head(X[tei]), -1); pred_v = (pv[:, 2] - pv[:, 0]).cpu().numpy()
        sf_v = (Y[tei][:, 2] - Y[tei][:, 0]).cpu().numpy()
        r = float(np.corrcoef(pred_v, sf_v)[0, 1])
    print(f"held-out soft-CE {vl:.3f} | value corr(pred, stockfish) r={r:.3f}")
    torch.save({"value_head": head.state_dict(), "d_model": d, "hidden": h, "arch": arch,
                "source_ckpt": a.ckpt}, a.out)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
