"""Residual style probe: freeze a baseline head H (trained on all non-novel train), then per
cluster train a regularized residual r_k so logits_k = H + r_k. r_k (zero-init) only has to model
the STYLE deviation, not the shared distribution -> cleaner signal.

Decisive metric (NO selection bias): for each cluster, does (H + r_k) beat H on that cluster's
HELD-OUT positions? Fixed assignment, held-out eval. Plus direct readouts: residual magnitude
(||r_k|| vs ||H||) and cross-cluster divergence of r_k on shared boards.
Runs on the cached frozen-encoder features (no encoder). Batch 1024 to fit the 4GB GPU.
"""
from __future__ import annotations
import argparse, numpy as np, torch, h5py
from style_policy.band_head import BandHead
from style_policy.legal_mask import u64_to_mask
from style_policy.loss import masked_square_ce

_NEG = float("-inf")
BS = 1024


def _read(f, idx):  # idx pre-sorted; keep squares/cls as float16 on CPU (RAM), cast per-batch
    return dict(
        cls=torch.from_numpy(f["cls"][idx]),            # f16
        sq=torch.from_numpy(f["squares"][idx]),         # f16
        fr=torch.from_numpy(f["from_sq"][idx].astype(np.int64)),
        to=torch.from_numpy(f["to_sq"][idx].astype(np.int64)),
        fm=u64_to_mask(torch.from_numpy(f["from_legal_u64"][idx].astype(np.uint64).astype(np.int64))),
        tm=u64_to_mask(torch.from_numpy(f["to_legal_u64"][idx].astype(np.uint64).astype(np.int64))),
    )


def _zero_last_linear(mod):
    last = None
    for m in mod.modules():
        if isinstance(m, torch.nn.Linear):
            last = m
    if last is not None:
        torch.nn.init.zeros_(last.weight)
        if last.bias is not None:
            torch.nn.init.zeros_(last.bias)


def _batches(n, shuffle=False, seed=0):
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed)) if shuffle else torch.arange(n)
    for i in range(0, n, BS):
        yield order[i:i + BS]


def _fl_tl(head, D, b, dev, base=None):
    """from/to logits of head on batch b, optionally added to a frozen base head's logits."""
    sq = D["sq"][b].to(dev).float(); cls = D["cls"][b].to(dev).float(); fr = D["fr"][b].to(dev)
    fl = head.from_logits(sq, cls); tl = head.to_logits(sq, fr, cls)
    if base is not None:
        with torch.no_grad():  # frozen base logits = constants
            bfl = base.from_logits(sq, cls); btl = base.to_logits(sq, fr, cls)
        fl = fl + bfl; tl = tl + btl  # add OUTSIDE no_grad so head keeps its grad
    return fl, tl


def train_head(D, rows, d, h, dev, epochs=2, wd=0.0, base=None, seed=0):
    head = BandHead(d, h, use_cls=True).to(dev)
    if base is not None:  # residual: start as no-op
        _zero_last_linear(head.from_head); _zero_last_linear(head.to_head)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=wd)
    sub = {k: v[rows] for k, v in D.items()}
    head.train()
    for ep in range(epochs):
        for b in _batches(len(rows), shuffle=True, seed=seed * 10 + ep):
            fl, tl = _fl_tl(head, sub, b, dev, base=base)
            loss = masked_square_ce(fl, sub["fr"][b].to(dev), sub["fm"][b].to(dev)) + \
                   masked_square_ce(tl, sub["to"][b].to(dev), sub["tm"][b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    # rebind sub into D-style dict for eval reuse
    return head, sub


@torch.no_grad()
def eval_mm_ce(D, rows, dev, head, base=None):
    """move-match% and mean (from+to) CE over rows, for head (optionally added to base)."""
    sub = {k: v[rows] for k, v in D.items()}
    hit = 0; ce = 0.0; n = len(rows)
    for b in _batches(n):
        fl, tl = _fl_tl(head, sub, b, dev, base=base)
        flm = fl.masked_fill(~sub["fm"][b].to(dev), _NEG); tlm = tl.masked_fill(~sub["tm"][b].to(dev), _NEG)
        pf = flm.argmax(-1); pt = tlm.argmax(-1)
        frb = sub["fr"][b].to(dev); tob = sub["to"][b].to(dev)
        hit += ((pf == frb) & (pt == tob)).sum().item()
        ce += (torch.nn.functional.cross_entropy(flm, frb, reduction="sum")
               + torch.nn.functional.cross_entropy(tlm, tob, reduction="sum")).item()
    return 100.0 * hit / n, ce / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/mnt/eloquence_bulk/databases/style_feat_cache.h5")
    ap.add_argument("--assign", default="/workspaces/eloquent-encoding/style_clustering_out/assignment.npz")
    ap.add_argument("--test-per-unit", type=int, default=40)
    ap.add_argument("--baseline-train", type=int, default=120000)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); dev = a.device; rng = np.random.default_rng(a.seed)

    A = np.load(a.assign, allow_pickle=True)
    u2c = {int(u): int(c) for u, c, nv in zip(A["unit_id"], A["cluster"], A["is_novel"]) if not nv}
    nonnovel = set(u2c)
    f = h5py.File(a.cache, "r")
    uid = f["unit_id"][:]; split = f["split"][:]
    test_by_unit = {}; train_pool = []
    for r in range(uid.shape[0]):
        u = int(uid[r])
        if u not in nonnovel: continue
        (test_by_unit.setdefault(u, []).append(r) if split[r] == 1 else train_pool.append(r))
    eval_rows = []; eval_cl = []
    for u, rows in test_by_unit.items():
        for r in rows[:a.test_per_unit]:
            eval_rows.append(r); eval_cl.append(u2c[u])
    eval_rows = np.array(eval_rows); eval_cl = np.array(eval_cl)
    tr = rng.choice(np.array(train_pool), min(a.baseline_train, len(train_pool)), replace=False)
    tr_cl = np.array([u2c[int(uid[r])] for r in tr])
    # sort each set by row for h5 read; keep companion arrays aligned
    eo = np.argsort(eval_rows); eval_rows, eval_cl = eval_rows[eo], eval_cl[eo]
    to = np.argsort(tr); tr, tr_cl = tr[to], tr_cl[to]
    print(f"eval {len(eval_rows):,} / train {len(tr):,} | clusters {sorted(set(u2c.values()))}", flush=True)

    E = _read(f, eval_rows); T = _read(f, tr)
    d, h = 384, 512

    # baseline H on all train
    print("training baseline H ...", flush=True)
    H, _ = train_head(T, np.arange(len(tr)), d, h, dev, epochs=2, wd=0.0, seed=0)

    # per-cluster residuals
    K = max(u2c.values()) + 1
    print(f"training {K} residuals (wd={a.wd}) ...", flush=True)
    resid = {}
    for k in range(K):
        rows_k = np.where(tr_cl == k)[0]
        if len(rows_k) < 100:
            resid[k] = None; continue
        resid[k], _ = train_head(T, rows_k, d, h, dev, epochs=2, wd=a.wd, base=H, seed=k + 1)

    # DECISIVE: per-cluster held-out H vs H+r_k
    print("\n===== RESIDUAL HELD-OUT TEST (H vs H+r_k on each cluster's test positions) =====")
    tot_h = 0.0; tot_hr = 0.0; tot_n = 0
    for k in range(K):
        rows_k = np.where(eval_cl == k)[0]
        if len(rows_k) == 0 or resid[k] is None:
            print(f"  cluster {k}: (skipped)"); continue
        mm_h, ce_h = eval_mm_ce(E, rows_k, dev, H)
        mm_hr, ce_hr = eval_mm_ce(E, rows_k, dev, resid[k], base=H)
        print(f"  cluster {k} (n={len(rows_k):5d}): move% H={mm_h:.2f} -> H+r={mm_hr:.2f} ({mm_hr-mm_h:+.2f}) | "
              f"CE H={ce_h:.4f} -> H+r={ce_hr:.4f} ({ce_hr-ce_h:+.4f})")
        tot_h += mm_h * len(rows_k); tot_hr += mm_hr * len(rows_k); tot_n += len(rows_k)
    print(f"  WEIGHTED: move% H={tot_h/tot_n:.2f} -> H+r={tot_hr/tot_n:.2f}  (delta {tot_hr/tot_n - tot_h/tot_n:+.3f})")

    # residual magnitude + cross-cluster divergence on a shared eval batch
    print("\n===== RESIDUAL MAGNITUDE + CROSS-CLUSTER DIVERGENCE =====")
    with torch.no_grad():
        b = torch.arange(min(4096, len(eval_rows)))
        sq = E["sq"][b].to(dev).float(); cls = E["cls"][b].to(dev).float()
        Hf = H.from_logits(sq, cls)
        rfs = [resid[k].from_logits(sq, cls) if resid[k] is not None else None for k in range(K)]
        Hn = Hf.norm(dim=-1).mean().item()
        for k in range(K):
            if rfs[k] is None: continue
            print(f"  cluster {k}: mean ||r_from|| = {rfs[k].norm(dim=-1).mean().item():.3f}  (||H_from|| = {Hn:.3f})")
        valid = [k for k in range(K) if rfs[k] is not None]
        if len(valid) >= 2:
            # mean pairwise L2 between clusters' residual from-logits (same boards)
            import itertools
            dists = [ (rfs[i]-rfs[j]).norm(dim=-1).mean().item() for i,j in itertools.combinations(valid,2) ]
            print(f"  mean pairwise ||r_i - r_j|| across clusters (same boards): {np.mean(dists):.3f}")
            # how often does argmax(H+r_k) differ across clusters?
            am = torch.stack([(Hf + rfs[k]).argmax(-1) for k in valid])  # (Kv, B)
            frac_diff = (am != am[0]).any(0).float().mean().item()
            print(f"  %% boards where top-from move differs across clusters: {100*frac_diff:.1f}%")


if __name__ == "__main__":
    main()
