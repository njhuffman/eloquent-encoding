"""Move-match diagnostic: on held-out human band-1500 positions, how often / how confidently does
the policy play the ACTUAL human move? Direct measurement (no discriminator). CE-on-human-moves is
the max-likelihood human-move objective, so the predictor is optimal on human states by construction;
this checks whether RWR changed move-match and where the ceiling is (vs the wider 128M-big encoder)."""
from __future__ import annotations
import argparse, sys, os, numpy as np, torch, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_policy.multiband_policy import MultiBandPolicy
from style_policy.legal_mask import u64_to_mask

DEV = "cuda"


def evaluate(ckpt, packed, hfrom, hto, fmask, tmask, band, bs=512):
    ck = torch.load(ckpt, map_location=DEV)
    m = MultiBandPolicy.from_config(ck["architecture"]); m.load_state_dict(ck["model"], strict=False)
    m.to(DEV).eval()
    head = m.heads[int(m.head_index(torch.tensor([band])).item())]
    fm = tm = 0.0; ce = 0.0; n = len(packed)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, n, bs):
            pk = torch.from_numpy(packed[i:i+bs].astype(np.int64)).to(DEV)
            cls, sq = m.encode(pk, hist=None)
            hf = hfrom[i:i+bs].to(DEV); ht = hto[i:i+bs].to(DEV)
            fmk = fmask[i:i+bs].to(DEV); tmk = tmask[i:i+bs].to(DEV)
            fl = head.from_logits(sq, cls).float().masked_fill(~fmk, -1e9)
            tl = head.to_logits(sq, hf, cls).float().masked_fill(~tmk, -1e9)
            fm += (fl.argmax(1) == hf).sum().item()
            tm += (tl.argmax(1) == ht).sum().item()
            lpf = torch.log_softmax(fl, 1).gather(1, hf[:, None]).squeeze(1)
            lpt = torch.log_softmax(tl, 1).gather(1, ht[:, None]).squeeze(1)
            ce += -(lpf + lpt).sum().item()
    return fm/n, tm/n, ce/n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="/mnt/eloquence_bulk/databases/wdl_validation_2025_05.h5")
    ap.add_argument("--band", type=int, default=1500); ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--ckpts", nargs="+", default=[
        "style_policy_checkpoints/multiband_ourdistill/multiband_ourdistill.pt",
        "style_policy_checkpoints/rwr_1500.pt",
        "style_policy_checkpoints/multiband_history_128M_big/multiband_history_128M_big.pt"])
    a = ap.parse_args()
    f = h5py.File(a.val, "r"); elo = f["elo_to_move"][:]
    pool = np.where((elo >= a.band) & (elo < a.band + 100))[0]
    idx = np.sort(np.random.default_rng(0).choice(pool, min(a.n, len(pool)), replace=False))
    packed = f["packed_pre"][idx]
    hfrom = torch.from_numpy(f["from_sq"][idx].astype(np.int64))
    hto = torch.from_numpy(f["to_sq"][idx].astype(np.int64))
    fmask = u64_to_mask(torch.from_numpy(np.array(f["from_legal_u64"][idx], dtype=np.uint64)).to(torch.int64))
    tmask = u64_to_mask(torch.from_numpy(np.array(f["to_legal_u64"][idx], dtype=np.uint64)).to(torch.int64))
    print(f"===== MOVE-MATCH vs human (band {a.band}, n={len(idx):,} held-out positions) =====")
    print(f"{'model':<40}{'from%':>7}{'to|f%':>7}{'joint%':>8}{'CE':>8}")
    for ck in a.ckpts:
        fm, tm, ce = evaluate(ck, packed, hfrom, hto, fmask, tmask, a.band)
        name = ck.split('/')[-1].replace('.pt', '')
        print(f"{name:<40}{100*fm:>7.1f}{100*tm:>7.1f}{100*fm*tm:>8.1f}{ce:>8.3f}", flush=True)
    print("\n  joint% ~ P(play exact human move); CE = -log P(human move). Predictor is CE-optimal on")
    print("  human states by construction -> RWR can't beat it here; 128M-big = stronger-encoder ceiling.")


if __name__ == "__main__":
    main()
