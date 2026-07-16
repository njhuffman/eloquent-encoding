"""How often does the human-head top move == the SF-move-head top move (argmax over legal 1792)?
Broken down by elo band. Also: is the SF top move within the human head's top-3? Uses held-out val."""
from __future__ import annotations
import argparse, sys, os, numpy as np, torch, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_policy.flat_policy import FlatMultiTaskPolicy
from style_policy.board_encode import packed_to_board
from style_policy.model_spec import elo_to_bucket
from style_policy import move_index

NEG = -1e9

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="style_policy_checkpoints/flat_multitask_128M/flat_multitask_128M.pt")
    ap.add_argument("--val", default="/mnt/eloquence_bulk/databases/wdl_validation_2025_05.h5")
    ap.add_argument("--n", type=int, default=10000); ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--bands", type=int, nargs="+", default=[1200, 1500, 1800, 2100])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); dev = a.device if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=dev)
    m = FlatMultiTaskPolicy.from_config(ck["architecture"]); m.load_state_dict(ck["model"], strict=False)
    m.to(dev).eval(); n_elo = int(ck["architecture"]["n_elo_buckets"])

    f = h5py.File(a.val, "r"); tot = f["packed_pre"].shape[0]
    idx = np.sort(np.random.default_rng(0).choice(tot, a.n, replace=False))
    packed = f["packed_pre"][idx]
    print(f"reconstructing {a.n:,} boards + legal masks ...", flush=True)
    masks = np.stack([move_index.legal_index_mask(packed_to_board(p.astype(np.uint8))) for p in packed])
    nlegal = masks.sum(1)                                    # legal-move count per board

    agree = {b: 0 for b in a.bands}; in3 = {b: 0 for b in a.bands}; done = 0
    with torch.no_grad():
        for i in range(0, a.n, a.bs):
            pk = torch.from_numpy(packed[i:i+a.bs].astype(np.int64)).to(dev)
            msk = torch.from_numpy(masks[i:i+a.bs]).to(dev)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                cls, sq = m.encode(pk, hist=None)
                sf_idx = m.sf_move_logits(cls, sq).float().masked_fill(~msk, NEG).argmax(1)
                for b in a.bands:
                    eidx = elo_to_bucket(torch.full((len(pk),), b), n_elo).to(dev)
                    hl = m.human_logits(cls, sq, eidx).float().masked_fill(~msk, NEG)
                    h_idx = hl.argmax(1)
                    agree[b] += int((h_idx == sf_idx).sum())
                    top3 = hl.topk(3, dim=1).indices
                    in3[b] += int((top3 == sf_idx[:, None]).any(1).sum())
            done += len(pk)
    print(f"\n===== HUMAN-head top move == SF-move-head top move (n={a.n:,}, mean legal={nlegal.mean():.1f}) =====")
    print(f"{'elo band':<10}{'top-1 agree':>13}{'SF-top in human top-3':>24}")
    for b in a.bands:
        print(f"{b:<10}{100*agree[b]/done:>12.1f}%{100*in3[b]/done:>23.1f}%", flush=True)


if __name__ == "__main__":
    main()
