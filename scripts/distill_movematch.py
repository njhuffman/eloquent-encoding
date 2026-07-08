"""Held-out human move-match for a MultiBandPolicy ckpt on 2025-05 (bands 1000-1900, in-range for
the distill experiment). Joint argmax vs the true human move. Run per encoder to compare."""
import argparse, sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_maia3_vs_ours import sample_positions, our_top1, _load_our_model, DEFAULT_PGN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pgn", default=DEFAULT_PGN)
    ap.add_argument("--per-band", type=int, default=400)
    ap.add_argument("--min-ply", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    samples, _ = sample_positions(a.pgn, a.per_band, a.min_ply, a.seed)
    model, arch, n_ply = _load_our_model(a.ckpt)
    hit = 0; n = 0; per = {}
    for s in samples:
        if s["self_elo"] >= 2000:  # encoders trained only on [1000,2000)
            continue
        pred = our_top1(model, n_ply, s["board"], s["self_elo"])
        ok = int(pred == s["actual"]); hit += ok; n += 1
        per.setdefault(s["band"], [0, 0]); per[s["band"]][0] += ok; per[s["band"]][1] += 1
    print(f"{a.ckpt.split('/')[-1]}: held-out move-match = {100*hit/n:.2f}%  (n={n})")
    for b in sorted(per):
        h, t = per[b]; print(f"    band {b}: {100*h/t:.1f}% (n={t})")


if __name__ == "__main__":
    main()
