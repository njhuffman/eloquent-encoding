"""Temperature sweep: for each elo band, vary self-play temperature and track blunder FREQUENCY
(P(Δeval < -0.2)) and blunder SCALE (mean |Δeval| given a blunder), vs the fixed human target.
Question: is there a T where self-play matches human blunder frequency, and does the scale still
match there — or does matching frequency require over-inflating scale?"""
from __future__ import annotations
import argparse, sys, os, numpy as np, h5py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import game_deltas as gd

TH = -0.20


def stats(d):
    bl = d < TH
    return 100 * bl.mean(), (float(-d[bl].mean()) if bl.any() else 0.0)     # freq%, conditional scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="style_policy_checkpoints/flat_multitask_128M/flat_multitask_128M.pt")
    ap.add_argument("--data", default="/mnt/eloquence_bulk/databases/wdl_history_128M.h5")
    ap.add_argument("--bands", type=int, nargs="+", default=[1000, 1500, 1900])
    ap.add_argument("--temps", type=float, nargs="+", default=[0.8, 1.0, 1.4, 1.8, 2.4])
    ap.add_argument("--n-human", type=int, default=6000); ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--max-plies", type=int, default=140)
    ap.add_argument("--out", default="/workspaces/eloquent-encoding/temp_sweep.png")
    a = ap.parse_args()
    mo = gd.M(a.ckpt)
    data = h5py.File(a.data, "r"); print("loading elo column ...", flush=True); elo_all = data["elo_to_move"][:]
    res = {}
    for band in a.bands:
        he, _ = gd.human_deltas(mo, data, elo_all, band, a.n_human)
        hf, hs = stats(he)
        res[band] = dict(hf=hf, hs=hs, T=[], f=[], s=[])
        print(f"\nband {band}: HUMAN blunder-freq {hf:.1f}%  blunder-scale {hs:.3f}", flush=True)
        for T in a.temps:
            se, _ = gd.selfplay_deltas(mo, band, a.games, a.max_plies, T)
            f, s = stats(se)
            res[band]["T"].append(T); res[band]["f"].append(f); res[band]["s"].append(s)
            print(f"  T={T:.1f}: self blunder-freq {f:5.1f}%  scale {s:.3f}  (n={len(se)})", flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {1000: "#d62728", 1500: "#2ca02c", 1900: "#1f77b4"}
    for band in a.bands:
        r = res[band]; c = colors.get(band, "k")
        ax[0].plot(r["T"], r["f"], "-o", color=c, label=f"self @{band}")
        ax[0].axhline(r["hf"], color=c, ls="--", alpha=0.7)
        ax[1].plot(r["T"], r["s"], "-o", color=c, label=f"self @{band}")
        ax[1].axhline(r["hs"], color=c, ls="--", alpha=0.7)
    ax[0].set_xlabel("self-play temperature"); ax[0].set_ylabel("blunder frequency %  (Δeval<-0.2)")
    ax[0].set_title("FREQUENCY vs T (dashed = human target)"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("self-play temperature"); ax[1].set_ylabel("blunder scale  (mean |Δeval| | blunder)")
    ax[1].set_title("SCALE vs T (dashed = human target)"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(a.out, dpi=110); print(f"\nsaved {a.out}", flush=True)


if __name__ == "__main__":
    main()
