"""Did style-clustering actually improve move prediction?

Compares top-1 move-match (argmax-from AND argmax-to|true-from, teacher-forced) on held-out
TEST positions of non-novel units, three ways:
  baseline  : ONE head trained on all (sampled) non-novel TRAIN positions (no clustering)
  clustered : each unit's positions scored by its ASSIGNED cluster head
  oracle    : each unit scored by its best-of-K head (upper bound; perfect assignment)
If clustered <= baseline, clustering bought no prediction progress. Oracle bounds the ceiling.
Runs on the cached frozen-encoder features (no encoder).
"""
from __future__ import annotations
import argparse, numpy as np, torch, h5py
from style_policy.band_head import BandHead
from style_policy.legal_mask import u64_to_mask

_NEG = float("-inf")


def _read_rows(f, idx):
    # idx MUST be pre-sorted ascending (h5py fancy indexing + caller alignment)
    return (
        torch.from_numpy(f["cls"][idx].astype(np.float32)),
        torch.from_numpy(f["squares"][idx].astype(np.float32)),
        torch.from_numpy(f["from_sq"][idx].astype(np.int64)),
        torch.from_numpy(f["to_sq"][idx].astype(np.int64)),
        torch.from_numpy(f["from_legal_u64"][idx].astype(np.uint64).astype(np.int64)),
        torch.from_numpy(f["to_legal_u64"][idx].astype(np.uint64).astype(np.int64)),
    )


@torch.no_grad()
def movematch(head, cls, sq, fr, to, fmask, tmask, dev, bs=1024):
    hit = 0; n = cls.shape[0]
    for i in range(0, n, bs):
        s = slice(i, i + bs)
        fl = head.from_logits(sq[s].to(dev), cls[s].to(dev)).masked_fill(~fmask[s].to(dev), _NEG)
        pf = fl.argmax(-1)
        tl = head.to_logits(sq[s].to(dev), fr[s].to(dev), cls[s].to(dev)).masked_fill(~tmask[s].to(dev), _NEG)
        pt = tl.argmax(-1)
        hit += ((pf == fr[s].to(dev)) & (pt == to[s].to(dev))).sum().item()
    return 100.0 * hit / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/mnt/eloquence_bulk/databases/style_feat_cache.h5")
    ap.add_argument("--assign", default="/workspaces/eloquent-encoding/style_clustering_out/assignment.npz")
    ap.add_argument("--heads", default="/workspaces/eloquent-encoding/style_clustering_out/heads.pt")
    ap.add_argument("--test-per-unit", type=int, default=40)
    ap.add_argument("--baseline-train", type=int, default=120000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); dev = a.device; rng = np.random.default_rng(a.seed)

    A = np.load(a.assign, allow_pickle=True)
    uid_arr, cluster, is_novel = A["unit_id"], A["cluster"], A["is_novel"]
    u2c = {int(u): int(c) for u, c, nv in zip(uid_arr, cluster, is_novel) if not nv}
    nonnovel = set(u2c)

    f = h5py.File(a.cache, "r")
    all_uid = f["unit_id"][:]; all_split = f["split"][:]
    # eval rows: up to test-per-unit TEST rows per non-novel unit
    eval_rows, eval_ucl = [], []
    train_pool = []
    by_unit_test = {}
    for r in range(all_uid.shape[0]):
        u = int(all_uid[r])
        if u not in nonnovel:
            continue
        if all_split[r] == 1:
            by_unit_test.setdefault(u, []).append(r)
        else:
            train_pool.append(r)
    for u, rows in by_unit_test.items():
        take = rows[:a.test_per_unit]
        eval_rows.extend(take); eval_ucl.extend([u2c[u]] * len(take))
    eval_rows = np.array(eval_rows); eval_ucl = np.array(eval_ucl)
    eval_unit = all_uid[eval_rows]
    # sort all three by row index so features (read sorted) stay aligned to cluster/unit
    o = np.argsort(eval_rows)
    eval_rows, eval_ucl, eval_unit = eval_rows[o], eval_ucl[o], eval_unit[o]
    tr = np.array(train_pool); tr = np.sort(rng.choice(tr, min(a.baseline_train, len(tr)), replace=False))
    print(f"eval positions: {len(eval_rows):,} over {len(by_unit_test):,} units | baseline-train: {len(tr):,}", flush=True)

    # load heads
    hd = torch.load(a.heads, map_location=dev); K = hd["k"]; d, h = hd["d_model"], hd["hidden"]
    heads = []
    for sd in hd["heads"]:
        m = BandHead(d, h, use_cls=True).to(dev).eval(); m.load_state_dict(sd); heads.append(m)

    # eval_rows already sorted above; features read in that order stay aligned to eval_ucl/eval_unit
    cls_e, sq_e, fr_e, to_e, fl_e, tl_e = _read_rows(f, eval_rows)
    eval_ucl_s = eval_ucl; eval_unit_s = eval_unit
    fmask_e = u64_to_mask(fl_e); tmask_e = u64_to_mask(tl_e)

    # train baseline head on tr
    clsT, sqT, frT, toT, flT, tlT = _read_rows(f, tr)
    fmaskT = u64_to_mask(flT); tmaskT = u64_to_mask(tlT)
    base = BandHead(d, h, use_cls=True).to(dev).train()
    opt = torch.optim.AdamW(base.parameters(), lr=1e-3)
    from style_policy.loss import masked_square_ce
    nT = clsT.shape[0]
    for ep in range(2):
        perm = torch.randperm(nT)
        for i in range(0, nT, 1024):
            b = perm[i:i+1024]
            fl = base.from_logits(sqT[b].to(dev), clsT[b].to(dev))
            tl = base.to_logits(sqT[b].to(dev), frT[b].to(dev), clsT[b].to(dev))
            loss = masked_square_ce(fl, frT[b].to(dev), fmaskT[b].to(dev)) + \
                   masked_square_ce(tl, toT[b].to(dev), tmaskT[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
    base.eval()

    # baseline move-match (overall)
    mm_base = movematch(base, cls_e, sq_e, fr_e, to_e, fmask_e, tmask_e, dev)

    # per-head move-match per position (for clustered + oracle)
    @torch.no_grad()
    def per_pos_hits(head):
        hits = np.zeros(cls_e.shape[0], dtype=bool)
        bs = 1024
        for i in range(0, cls_e.shape[0], bs):
            s = slice(i, i+bs)
            fl = head.from_logits(sq_e[s].to(dev), cls_e[s].to(dev)).masked_fill(~fmask_e[s].to(dev), _NEG)
            pf = fl.argmax(-1)
            tl = head.to_logits(sq_e[s].to(dev), fr_e[s].to(dev), cls_e[s].to(dev)).masked_fill(~tmask_e[s].to(dev), _NEG)
            pt = tl.argmax(-1)
            hits[s] = ((pf == fr_e[s].to(dev)) & (pt == to_e[s].to(dev))).cpu().numpy()
        return hits
    head_hits = np.stack([per_pos_hits(hh) for hh in heads])  # (K, Npos)

    # clustered: each position scored by its unit's assigned cluster head
    clustered = head_hits[eval_ucl_s, np.arange(head_hits.shape[1])].mean() * 100
    # oracle: per unit, pick the head with best move-match on that unit's eval rows
    oracle_hits = 0; oracle_n = 0
    for u in np.unique(eval_unit_s):
        m = eval_unit_s == u
        best = head_hits[:, m].mean(axis=1).argmax()
        oracle_hits += head_hits[best, m].sum(); oracle_n += m.sum()
    oracle = 100.0 * oracle_hits / oracle_n

    print("\n===== MOVE-MATCH: did clustering help predict held-out moves? =====")
    print(f"  baseline  (1 head, no clustering) : {mm_base:.2f}%")
    print(f"  clustered (assigned head)         : {clustered:.2f}%   (delta vs baseline: {clustered-mm_base:+.2f})")
    print(f"  oracle    (best-of-K per unit)    : {oracle:.2f}%   (delta vs baseline: {oracle-mm_base:+.2f})  [upper bound]")


if __name__ == "__main__":
    main()
