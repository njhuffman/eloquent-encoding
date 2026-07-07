#!/usr/bin/env python
"""K=4 EM style-clustering on CACHED frozen-encoder features (no encoder here).

Clusters (player,color) *units* into K groups by *which small move-prediction head
best predicts them*. Each cluster owns a fresh ``BandHead`` (use_cls=True) trained on
the cached (cls, squares) features of its currently-assigned units. EM alternates:

  E-step: re-init a fresh BandHead per cluster, train it on the cluster's TRAIN positions.
  M-step: reassign every non-novel unit to the head with the lowest mean move-CE on the
          unit's TRAIN positions.

Move-CE of a head on a position = masked_square_ce(from) + masked_square_ce(to), with the
to-head teacher-forced on the true from-square (same metric as the K-sweep). Everything
runs on the frozen features already cached in style_feat_cache.h5 -- the encoder is never
loaded. Only the huge ``squares`` array is streamed from disk per pass; all small columns
and ``cls`` are held in RAM.

Metrics printed at the end (the payoff): convergence/sizes, held-out divergence,
novel-player test, W/B consistency, and per-cluster elo+color characterization.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
import h5py
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from style_policy.band_head import BandHead
from style_policy.loss import masked_square_ce
from style_policy.legal_mask import u64_to_mask

_NEG = float("-inf")


# ----------------------------------------------------------------------------- helpers
def log(msg):
    print(msg, flush=True)


def move_ce_per_row(head, sq, cls, fsq, tsq, fmask, tmask, use_amp):
    """Per-position move-CE (from-CE + to-CE, teacher-forced to) as a (N,) tensor.

    Same masked-softmax formulation as style_policy.loss.masked_square_ce, but returns
    the per-row value (that fn returns the batch mean) so we can accumulate per unit.
    """
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        fl = head.from_logits(sq, cls)
        tl = head.to_logits(sq, fsq, cls)

    def _perrow(logits, target, mask):
        logits = logits.float().masked_fill(~mask, _NEG)
        logp = torch.log_softmax(logits, dim=-1)
        return -logp.gather(1, target[:, None]).squeeze(1)

    return _perrow(fl, fsq, fmask) + _perrow(tl, tsq, tmask)


class Cache:
    """Holds all small columns + cls in RAM; streams squares from the open h5."""

    def __init__(self, h5_path):
        self.f = h5py.File(h5_path, "r")
        self.sq = self.f["squares"]                       # (M,64,384) f16 -- streamed
        self.M = self.sq.shape[0]
        self.d_model = self.sq.shape[2]
        log(f"  loading small columns + cls into RAM (M={self.M}) ...")
        self.unit_id = self.f["unit_id"][:].astype(np.int64)
        self.split = self.f["split"][:].astype(np.int8)
        self.from_sq = self.f["from_sq"][:].astype(np.int64)
        self.to_sq = self.f["to_sq"][:].astype(np.int64)
        # reinterpret uint64 bitboards as int64 bit-for-bit (values may exceed int64 max)
        self.from_legal = self.f["from_legal_u64"][:].view(np.int64)
        self.to_legal = self.f["to_legal_u64"][:].view(np.int64)
        self.cls = self.f["cls"][:]                        # (M,384) f16 in RAM (~0.85 GB)

    def chunks(self, read_chunk):
        for s in range(0, self.M, read_chunk):
            e = min(s + read_chunk, self.M)
            yield s, e, self.sq[s:e]                        # numpy (b,64,384) f16


def to_gpu_batch(cache, sq_chunk, chunk_start, local_idx, device):
    """Build GPU tensors for a set of local indices within a streamed chunk."""
    g = chunk_start + local_idx                             # global row indices
    sq = torch.from_numpy(np.ascontiguousarray(sq_chunk[local_idx])).to(device).float()
    cls = torch.from_numpy(np.ascontiguousarray(cache.cls[g])).to(device).float()
    fsq = torch.from_numpy(cache.from_sq[g]).to(device)
    tsq = torch.from_numpy(cache.to_sq[g]).to(device)
    fmask = u64_to_mask(torch.from_numpy(cache.from_legal[g]).to(device))
    tmask = u64_to_mask(torch.from_numpy(cache.to_legal[g]).to(device))
    return sq, cls, fsq, tsq, fmask, tmask


# ----------------------------------------------------------------------------- E-step
def e_step(cache, assign, row_unit, is_train, K, d_model, hidden, epochs, lr,
           read_chunk, device, seed, it, max_gpu):
    heads, opts = [], []
    for k in range(K):
        torch.manual_seed(seed * 1000 + it * 10 + k)        # deterministic fresh init
        h = BandHead(d_model=d_model, hidden=hidden, use_cls=True).to(device)
        h.train()
        heads.append(h)
        opts.append(torch.optim.AdamW(h.parameters(), lr=lr))
    use_amp = device == "cuda"
    row_cluster = assign[row_unit]                          # (M,) cluster per row (-1 novel)
    for epoch in range(epochs):
        for s, e, sq_chunk in cache.chunks(read_chunk):
            cl = row_cluster[s:e]
            tr = is_train[s:e]
            for k in range(K):
                sel = np.where((cl == k) & tr)[0]
                if sel.size == 0:
                    continue
                # sub-batch to bound GPU memory (batch ~ read_chunk*train_frac/K)
                for b0 in range(0, sel.size, max_gpu):
                    sub = sel[b0:b0 + max_gpu]
                    sqb, clsb, fsq, tsq, fmask, tmask = to_gpu_batch(cache, sq_chunk, s, sub, device)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                        fl = heads[k].from_logits(sqb, clsb)
                        tl = heads[k].to_logits(sqb, fsq, clsb)
                        loss = (masked_square_ce(fl, fsq, fmask)
                                + masked_square_ce(tl, tsq, tmask))
                    opts[k].zero_grad(set_to_none=True)
                    loss.backward()
                    opts[k].step()
    for h in heads:
        h.eval()
    return heads


# --------------------------------------------------------------------- CE accumulation
def accumulate_ce(cache, heads, row_unit, row_mask, K, n_units, read_chunk, device, max_gpu):
    """Mean move-CE per (unit, head) over the rows selected by row_mask (bool, len M)."""
    ce_sum = np.zeros((n_units, K), dtype=np.float64)
    cnt = np.zeros(n_units, dtype=np.int64)
    use_amp = device == "cuda"
    for s, e, sq_chunk in cache.chunks(read_chunk):
        m = row_mask[s:e]
        sel_all = np.where(m)[0]
        if sel_all.size == 0:
            continue
        for b0 in range(0, sel_all.size, max_gpu):
            sub = sel_all[b0:b0 + max_gpu]
            units = row_unit[s + sub]
            np.add.at(cnt, units, 1)
            sqb, clsb, fsq, tsq, fmask, tmask = to_gpu_batch(cache, sq_chunk, s, sub, device)
            for k in range(K):
                pr = move_ce_per_row(heads[k], sqb, clsb, fsq, tsq, fmask, tmask, use_amp)
                np.add.at(ce_sum[:, k], units, pr.float().cpu().numpy())
    mean = np.full((n_units, K), np.nan, dtype=np.float64)
    nz = cnt > 0
    mean[nz] = ce_sum[nz] / cnt[nz, None]
    return mean, cnt


# ----------------------------------------------------------------------------- metrics
def pct(x):
    return 100.0 * float(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/mnt/eloquence_bulk/databases/style_feat_cache.h5")
    ap.add_argument("--units", default="/mnt/eloquence_bulk/databases/style_feat_cache_units.npz")
    ap.add_argument("--out-dir", default="style_clustering_out")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-iters", type=int, default=8)
    ap.add_argument("--head-epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--read-chunk", type=int, default=16384)
    ap.add_argument("--max-gpu", type=int, default=4096, help="max rows per GPU forward")
    ap.add_argument("--stop-frac", type=float, default=0.01)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-rows", type=int, default=0, help="debug: truncate M to first N rows")
    args = ap.parse_args()

    K = args.k
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    t_all = time.time()

    log(f"[load] cache={args.cache}")
    cache = Cache(args.cache)
    if args.limit_rows:
        cache.M = min(args.limit_rows, cache.M)
        for a in ("unit_id", "split", "from_sq", "to_sq", "from_legal", "to_legal"):
            setattr(cache, a, getattr(cache, a)[:cache.M])
        cache.cls = cache.cls[:cache.M]
    d_model = cache.d_model

    z = np.load(args.units, allow_pickle=True)
    u_ids = z["unit_id"].astype(np.int64)
    usernames = z["username"]
    colors = z["color"].astype(np.int64)          # 0=W, 1=B
    elo_mean = z["elo_mean"].astype(np.float64)
    is_novel = z["is_novel"].astype(bool)
    n_units = len(u_ids)
    nonnovel = ~is_novel
    n_nonnovel = int(nonnovel.sum())

    lut = np.full(int(cache.unit_id.max()) + 1, -1, dtype=np.int64)
    lut[u_ids] = np.arange(n_units)
    row_unit = lut[cache.unit_id]                  # (M,) unit index per row
    is_train = cache.split == 0
    is_test = cache.split == 1
    train_mask = is_train.copy()
    test_mask = is_test.copy()
    log(f"[load] M={cache.M} units={n_units} (nonnovel={n_nonnovel}, novel={int(is_novel.sum())}) "
        f"train_rows={int(is_train.sum())} test_rows={int(is_test.sum())} device={device}")

    # ---- init assignment: random cluster for each non-novel unit; novel = -1
    rng = np.random.default_rng(args.seed)
    assign = np.full(n_units, -1, dtype=np.int64)
    assign[nonnovel] = rng.integers(0, K, size=n_nonnovel)

    reassign_hist = []
    heads = None
    mean_ce_train = None
    for it in range(args.max_iters):
        t0 = time.time()
        heads = e_step(cache, assign, row_unit, is_train, K, d_model, args.hidden,
                       args.head_epochs, args.lr, args.read_chunk, device, args.seed, it, args.max_gpu)
        mean_ce_train, _ = accumulate_ce(cache, heads, row_unit, train_mask, K, n_units,
                                         args.read_chunk, device, args.max_gpu)
        # M-step: reassign non-novel units to argmin-CE head on TRAIN positions
        new_assign = assign.copy()
        nn_ce = mean_ce_train[nonnovel]
        new_assign[nonnovel] = np.argmin(nn_ce, axis=1)
        changed = int(np.sum(new_assign[nonnovel] != assign[nonnovel]))
        frac = changed / n_nonnovel
        reassign_hist.append(frac)
        sizes = np.bincount(new_assign[nonnovel], minlength=K)
        log(f"[iter {it}] reassigned {changed}/{n_nonnovel} ({pct(frac):.2f}%)  "
            f"sizes={sizes.tolist()}  ({time.time()-t0:.0f}s)")
        assign = new_assign
        if frac < args.stop_frac:
            log(f"[iter {it}] converged (<{pct(args.stop_frac):.0f}% changed)")
            break

    # ---- final held-out (TEST) CE for all units under the final heads
    log("[metrics] computing held-out TEST CE ...")
    mean_ce_test, cnt_test = accumulate_ce(cache, heads, row_unit, test_mask, K, n_units,
                                           args.read_chunk, device, args.max_gpu)

    # ================================================================= METRICS
    nn_idx = np.where(nonnovel)[0]
    nv_idx = np.where(is_novel)[0]
    assigned = assign                                  # final assignment (unit idx -> cluster)

    # --- 1. convergence & sizes
    sizes = np.bincount(assigned[nonnovel], minlength=K)
    size_frac = sizes / n_nonnovel
    collapse = [k for k in range(K) if size_frac[k] < 0.05 or size_frac[k] > 0.70]

    # --- 2. held-out divergence (TEST)  [use only units with finite TEST CE under all heads]
    tr_test_all = mean_ce_test[nn_idx]                  # (n_nn, K)
    fin2 = np.isfinite(tr_test_all).all(axis=1)
    tr_test = tr_test_all[fin2]
    asg_nn = assigned[nn_idx][fin2]
    n_heldout = int(fin2.sum())
    ce_assigned = tr_test[np.arange(n_heldout), asg_nn]
    masked = tr_test.copy()
    masked[np.arange(n_heldout), asg_nn] = np.inf
    ce_secondbest = masked.min(axis=1)                  # best competitor (not assigned)
    delta_test = ce_secondbest - ce_assigned            # >0 => assigned best on unseen games
    argmin_test = np.argmin(tr_test, axis=1)
    agree_test = float(np.mean(argmin_test == asg_nn))
    mean_delta_test = float(np.mean(delta_test))

    # --- 3. novel-player test
    # confidence gap = (2nd-smallest CE) - (smallest CE), on each unit's own positions
    def gap_of(ce_rows):
        fin = np.isfinite(ce_rows).all(axis=1)
        srt = np.sort(ce_rows[fin], axis=1)
        return srt[:, 1] - srt[:, 0]
    gap_nonnovel_train = gap_of(mean_ce_train[nn_idx])
    nn_median_gap = float(np.median(gap_nonnovel_train))

    nv_train_all = mean_ce_train[nv_idx]                # novel units never trained any head
    finv = np.isfinite(nv_train_all).all(axis=1)
    nv_train = nv_train_all[finv]
    nv_units = nv_idx[finv]
    nv_assign_train = np.argmin(nv_train, axis=1)
    nv_gap = nv_train.copy()
    srt = np.sort(nv_gap, axis=1)
    nv_gap = srt[:, 1] - srt[:, 0]
    nv_median_gap = float(np.median(nv_gap))
    nv_frac_above = float(np.mean(nv_gap > nn_median_gap))
    nv_sizes = np.bincount(nv_assign_train, minlength=K)
    nv_test = mean_ce_test[nv_units]
    finvt = np.isfinite(nv_test).all(axis=1)
    nv_argmin_test = np.argmin(nv_test[finvt], axis=1)
    nv_test_agree = float(np.mean(nv_argmin_test == nv_assign_train[finvt]))

    # --- 4. W/B consistency (players with both a W-unit and B-unit in non-novel set)
    nn_set = set(nn_idx.tolist())
    by_user = {}
    for i in nn_idx:
        by_user.setdefault(str(usernames[i]), {})[int(colors[i])] = int(assigned[i])
    both = {u: d for u, d in by_user.items() if 0 in d and 1 in d}
    same = sum(1 for d in both.values() if d[0] == d[1])
    wb_same_frac = (same / len(both)) if both else float("nan")

    # --- 5. characterization (elo + color per cluster)
    cluster_char = []
    for k in range(K):
        members = nn_idx[assigned[nn_idx] == k]
        if members.size == 0:
            cluster_char.append({"k": k, "n": 0, "elo_mean": None, "n_white": 0, "n_black": 0})
            continue
        cluster_char.append({
            "k": k,
            "n": int(members.size),
            "elo_mean": float(np.mean(elo_mean[members])),
            "elo_std": float(np.std(elo_mean[members])),
            "n_white": int(np.sum(colors[members] == 0)),
            "n_black": int(np.sum(colors[members] == 1)),
        })
    overall_elo = float(np.mean(elo_mean[nn_idx]))

    # ================================================================= REPORT
    log("")
    log("=" * 72)
    log(f"  K={K} EM STYLE-CLUSTERING REPORT  (seed={args.seed})")
    log("=" * 72)
    log("")
    log("[1] CONVERGENCE & SIZES")
    for i, f in enumerate(reassign_hist):
        log(f"    iter {i}: reassigned {pct(f):6.2f}%")
    log(f"    iterations run: {len(reassign_hist)}  (max {args.max_iters})")
    log(f"    final cluster sizes (non-novel, n={n_nonnovel}):")
    for k in range(K):
        log(f"      cluster {k}: {int(sizes[k]):5d} units ({pct(size_frac[k]):5.1f}%)")
    log(f"    collapse flag (any <5% or >70%): {'YES -> ' + str(collapse) if collapse else 'no'}")
    log("")
    log("[2] HELD-OUT DIVERGENCE (TEST positions, the key real-signal test)")
    log(f"    units with TEST coverage: {n_heldout}/{len(nn_idx)}")
    log(f"    mean over units of (CE_secondbest - CE_assigned) on TEST : {mean_delta_test:+.4f}")
    log(f"      (>0 => assigned head predicts UNSEEN games better than any other head)")
    log(f"    %% units whose ASSIGNED head is also argmin on TEST      : {pct(agree_test):5.1f}%  (chance=25%)")
    log("")
    log("[3] NOVEL-PLAYER TEST (units never in any head's training)")
    log(f"    novel units: {len(nv_idx)}   argmin-assignment sizes: {nv_sizes.tolist()}")
    log(f"    confidence gap (2nd-best - best CE) on their TRAIN positions:")
    log(f"      median novel gap        : {nv_median_gap:.4f}")
    log(f"      median non-novel gap    : {nn_median_gap:.4f}")
    log(f"      %% novel with gap > non-novel median : {pct(nv_frac_above):5.1f}%")
    log(f"    novel TEST argmin agreement with their TRAIN assignment : {pct(nv_test_agree):5.1f}%  (chance=25%)")
    log("")
    log("[4] W/B CONSISTENCY (players with both a White-unit and Black-unit, non-novel)")
    log(f"    such players: {len(both)}")
    log(f"    %% landing in the SAME cluster : "
        + (f"{pct(wb_same_frac):5.1f}%  (chance=25%)" if both else "n/a"))
    log(f"      (>25% => personal style spans color; ~25% => color-specific)")
    log("")
    log("[5] CHARACTERIZATION (is it style, not strength?)")
    log(f"    overall non-novel mean elo: {overall_elo:.1f}")
    log(f"    {'cluster':>8} {'n':>6} {'elo_mean':>10} {'elo_std':>9} {'#White':>7} {'#Black':>7}")
    for c in cluster_char:
        em = f"{c['elo_mean']:.1f}" if c["elo_mean"] is not None else "n/a"
        es = f"{c.get('elo_std', 0.0):.1f}" if c["elo_mean"] is not None else "n/a"
        log(f"    {c['k']:>8} {c['n']:>6} {em:>10} {es:>9} {c['n_white']:>7} {c['n_black']:>7}")
    elos = [c["elo_mean"] for c in cluster_char if c["elo_mean"] is not None]
    elo_spread = (max(elos) - min(elos)) if elos else 0.0
    log(f"    elo spread across clusters (max-min): {elo_spread:.1f}  "
        f"(small => NOT strength-driven)")
    log("")
    log(f"[done] total wall time {time.time()-t_all:.0f}s")
    log("=" * 72)

    # ================================================================= SAVE
    torch.save({"heads": [h.state_dict() for h in heads],
                "d_model": d_model, "hidden": args.hidden, "k": K},
               os.path.join(args.out_dir, "heads.pt"))
    np.savez(os.path.join(args.out_dir, "assignment.npz"),
             unit_id=u_ids, cluster=assigned, is_novel=is_novel,
             novel_assign_train=(lambda a: (a.__setitem__(nv_units, nv_assign_train) or a))(
                 np.full(n_units, -1, dtype=np.int64)),
             username=usernames, color=colors, elo_mean=elo_mean)
    metrics = {
        "k": K, "seed": args.seed,
        "iterations": len(reassign_hist),
        "reassign_frac_per_iter": reassign_hist,
        "final_sizes": sizes.tolist(),
        "final_size_frac": size_frac.tolist(),
        "collapse_clusters": collapse,
        "heldout_units": n_heldout,
        "heldout_mean_delta_test": mean_delta_test,
        "heldout_argmin_agreement": float(agree_test),
        "novel_median_gap": nv_median_gap,
        "nonnovel_median_gap": nn_median_gap,
        "novel_frac_gap_above_nonnovel_median": nv_frac_above,
        "novel_assign_sizes": nv_sizes.tolist(),
        "novel_test_argmin_agreement": nv_test_agree,
        "wb_players": len(both),
        "wb_same_cluster_frac": (None if not both else float(wb_same_frac)),
        "overall_nonnovel_elo": overall_elo,
        "cluster_char": cluster_char,
        "elo_spread_across_clusters": elo_spread,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    log(f"[save] heads.pt, assignment.npz, metrics.json -> {args.out_dir}/")


if __name__ == "__main__":
    main()
