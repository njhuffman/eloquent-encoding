"""Is the ~0.08 predictor-vs-human drift gap REAL headroom or a weak-discriminator artifact?
Roll out predictor@T from human ply-10 seeds to +24, then measure human-vs-bot distinguishability
with (1) the current WEAK disc on encoder feats, (2) a STRONG disc on encoder feats, (3) a STRONG
disc on RAW board planes. Report AUC and gap-over-floor (floor = same disc on human-A vs human-B)."""
from __future__ import annotations
import argparse, os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selfplay_drift import load, feats, disc_auc, auc
from selfplay_drift_seeded import collect_seeds, rollout
from concept_probe import board_planes
from style_policy.board_encode import board_to_packed, packed_to_board


def raw_feats(packed_arr):
    return torch.from_numpy(np.stack([board_planes(packed_to_board(p.astype(np.uint8)))
                                      for p in packed_arr])).float()


def strong_auc(Fpos, Fneg, dev, hidden=256, epochs=120, seed=0):
    """Big MLP, train to convergence, no early stop -> the discriminator's CEILING."""
    torch.manual_seed(seed)
    X = torch.cat([Fpos, Fneg]); Y = torch.cat([torch.ones(len(Fpos)), torch.zeros(len(Fneg))])
    p = torch.randperm(len(X)); ntr = int(0.8*len(X)); tri, tei = p[:ntr], p[ntr:]
    D = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(),
                            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                            torch.nn.Linear(hidden, 1)).to(dev)
    opt = torch.optim.AdamW(D.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        pp = tri[torch.randperm(len(tri))]
        for i in range(0, len(tri), 512):
            b = pp[i:i+512]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(D(X[b].to(dev)).squeeze(1), Y[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        s = D(X[tei].to(dev)).squeeze(1).cpu().numpy()
    return auc(s, Y[tei].numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="style_policy_checkpoints/multiband_ourdistill/multiband_ourdistill.pt")
    ap.add_argument("--pgn", default="/mnt/eloquence_bulk/databases/lichess_db_standard_rated_2025-05_tc_600_0.pgn.zst")
    ap.add_argument("--band", type=int, default=1500); ap.add_argument("--seed-ply", type=int, default=10)
    ap.add_argument("--n-seeds", type=int, default=1200); ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--device", default="cuda"); a = ap.parse_args(); dev = a.device
    model, n_ply = load(a.ckpt, dev)
    head = model.heads[int(model.head_index(torch.tensor([a.band])).item())]
    H = 24
    print(f"seeds (band {a.band}, ply {a.seed_ply}), predictor @ T={a.temp}, horizon +{H} ...", flush=True)
    seeds = collect_seeds(a.pgn, a.band, a.seed_ply, [H], a.n_seeds, n_ply)
    snaps = rollout(model, head, n_ply, seeds, [H], dev, temp=a.temp)
    human = np.stack([board_to_packed(s[2][H]) for s in seeds])
    bot = np.stack(snaps[H])
    print(f"  {len(seeds)} seeds; encoding ...", flush=True)
    Fh, Fb = feats(model, human, dev), feats(model, bot, dev)         # encoder [CLS++mean-sq]
    Rh, Rb = raw_feats(human), raw_feats(bot)                          # raw 8x8 planes
    nh = len(Fh)//2; nr = len(Rh)//2

    print(f"\n===== DRIFT CEILING (predictor@T={a.temp}, +{H} plies, n={len(seeds)}) =====")
    print(f"{'discriminator':<26}{'human-vs-bot':>13}{'floor(A/B)':>12}{'GAP':>8}")
    w, wf = disc_auc(Fh, Fb, dev), disc_auc(Fh[:nh], Fh[nh:], dev)
    print(f"{'weak (enc feats)':<26}{w:>13.3f}{wf:>12.3f}{w-wf:>8.3f}")
    se, sef = strong_auc(Fh, Fb, dev), strong_auc(Fh[:nh], Fh[nh:], dev)
    print(f"{'STRONG (enc feats)':<26}{se:>13.3f}{sef:>12.3f}{se-sef:>8.3f}")
    sr, srf = strong_auc(Rh, Rb, dev), strong_auc(Rh[:nr], Rh[nr:], dev)
    print(f"{'STRONG (RAW planes)':<26}{sr:>13.3f}{srf:>12.3f}{sr-srf:>8.3f}")
    print("\n  GAP = human-vs-bot AUC - floor. If STRONG/RAW gap >> weak gap => real headroom the weak")
    print("  reward-disc missed. If all gaps ~equal & small => predictor genuinely near-human (no headroom).")


if __name__ == "__main__":
    main()
