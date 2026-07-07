"""Group (player,color) units by an interpretable BEHAVIORAL style metric, instead of unsupervised
EM. First metric: immediate-recapture rate = of positions where the opponent just captured
(hist_cap[0]>0) and the player has a move, the fraction where the player recaptures on that same
square (to_sq == hist_to[0]). High => immediate recapturer; low => tension-holder/developer.

Writes an assignment npz (unit_id, cluster=quartile 0..K-1 for non-novel, is_novel) that
scripts/style_residual_check.py --assign can consume unchanged.
"""
from __future__ import annotations
import argparse, numpy as np, h5py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style-h5", default="/mnt/eloquence_bulk/databases/style_1500_2025_01.h5")
    ap.add_argument("--subset", default="/workspaces/eloquent-encoding/style_clustering_out/assignment.npz",
                    help="npz defining the subset units + is_novel (unit_id, is_novel[, username, color])")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--min-opp", type=int, default=10)
    ap.add_argument("--out", default="/workspaces/eloquent-encoding/style_clustering_out/metric_assignment.npz")
    a = ap.parse_args()

    S = np.load(a.subset, allow_pickle=True)
    sub_uid = S["unit_id"].astype(np.int64)
    is_novel = S["is_novel"].astype(bool)
    novel_map = {int(u): bool(nv) for u, nv in zip(sub_uid, is_novel)}
    sub_set = set(int(u) for u in sub_uid)

    f = h5py.File(a.style_h5, "r")
    uid = f["unit_id"][:].astype(np.int64)
    split = f["split"][:]
    to_sq = f["to_sq"][:].astype(np.int64)
    hc0 = f["hist_cap"][:, 0].astype(np.int64)   # cap type of the immediately-preceding ply
    ht0 = f["hist_to"][:, 0].astype(np.int64)    # to-square of that ply (where opp captured)

    in_sub = np.isin(uid, np.array(sorted(sub_set)))
    opp = (split == 0) & (hc0 > 0) & in_sub        # opponent just captured, on a train position
    recap = opp & (to_sq == ht0)                    # player took back on that square
    maxu = uid.max() + 1
    opp_ct = np.bincount(uid[opp], minlength=maxu).astype(np.float64)
    rec_ct = np.bincount(uid[recap], minlength=maxu).astype(np.float64)

    # per-unit rate for subset units
    rows = []
    for u in sorted(sub_set):
        o = opp_ct[u]; r = rec_ct[u]
        rows.append((u, novel_map[u], o, (r / o) if o > 0 else np.nan))
    uids = np.array([x[0] for x in rows], dtype=np.int64)
    nov = np.array([x[1] for x in rows], dtype=bool)
    opps = np.array([x[2] for x in rows], dtype=np.float64)
    rate = np.array([x[3] for x in rows], dtype=np.float64)

    # quartile-group NON-NOVEL units with enough opportunities, by rate
    elig = (~nov) & (opps >= a.min_opp) & np.isfinite(rate)
    r_elig = rate[elig]
    qs = np.quantile(r_elig, np.linspace(0, 1, a.k + 1))
    qs[0] -= 1e-9; qs[-1] += 1e-9
    cluster = np.full(len(uids), -1, dtype=np.int64)
    cluster[elig] = np.clip(np.digitize(rate[elig], qs) - 1, 0, a.k - 1)
    # non-novel units with too few opps: drop from clustering (cluster stays -1 -> residual probe skips)
    # novel units keep cluster -1 too (probe uses is_novel to hold them out).

    print(f"subset units: {len(uids)} | non-novel eligible (opp>={a.min_opp}): {int(elig.sum())}")
    print(f"recapture-rate: min={np.nanmin(r_elig):.3f} q25={np.quantile(r_elig,.25):.3f} "
          f"median={np.quantile(r_elig,.5):.3f} q75={np.quantile(r_elig,.75):.3f} max={np.nanmax(r_elig):.3f}")
    print(f"quartile edges: {np.round(qs,3)}")
    for k in range(a.k):
        m = cluster == k
        print(f"  group {k}: n={int(m.sum()):5d}  mean recapture-rate={rate[m].mean():.3f}  "
              f"mean opps={opps[m].mean():.0f}")

    # units that didn't get a cluster (novel, or too few opps) are held out from the probe
    nov_out = nov | (cluster == -1)
    np.savez(a.out, unit_id=uids, cluster=cluster, is_novel=nov_out,
             username=S["username"] if "username" in S else uids.astype(str),
             color=S["color"] if "color" in S else np.zeros(len(uids), np.int8),
             elo_mean=S["elo_mean"] if "elo_mean" in S else np.zeros(len(uids)),
             recapture_rate=rate, n_opp=opps)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
