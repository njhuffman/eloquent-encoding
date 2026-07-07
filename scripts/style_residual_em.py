"""Residual-EM seeded from a good (metric-based) grouping: iterate
  E: train K residuals r_k (logits = frozen H + r_k) on current groups' TRAIN
  M: reassign each unit to the residual with lowest CE on its TRAIN positions
measuring HELD-OUT (test) gain each iteration to detect overfitting/drift.
Question: do the clusters get *better* (held-out gain rises) or drift (falls)?
Runs on cached frozen-encoder features. Batch 1024 (4GB GPU).
"""
from __future__ import annotations
import argparse, numpy as np, torch, h5py
from style_policy.band_head import BandHead
from style_policy.legal_mask import u64_to_mask
from style_policy.loss import masked_square_ce

_NEG = float("-inf"); BS = 1024


def _read(f, idx):
    return dict(
        cls=torch.from_numpy(f["cls"][idx]), sq=torch.from_numpy(f["squares"][idx]),
        fr=torch.from_numpy(f["from_sq"][idx].astype(np.int64)),
        to=torch.from_numpy(f["to_sq"][idx].astype(np.int64)),
        fm=u64_to_mask(torch.from_numpy(f["from_legal_u64"][idx].astype(np.uint64).astype(np.int64))),
        tm=u64_to_mask(torch.from_numpy(f["to_legal_u64"][idx].astype(np.uint64).astype(np.int64))),
    )


def _zero_last(mod):
    last = [m for m in mod.modules() if isinstance(m, torch.nn.Linear)][-1]
    torch.nn.init.zeros_(last.weight);  last.bias is not None and torch.nn.init.zeros_(last.bias)


def _b(n, sh=False, seed=0):
    o = torch.randperm(n, generator=torch.Generator().manual_seed(seed)) if sh else torch.arange(n)
    for i in range(0, n, BS): yield o[i:i+BS]


def _logits(head, D, b, dev, base=None):
    sq = D["sq"][b].to(dev).float(); cls = D["cls"][b].to(dev).float(); fr = D["fr"][b].to(dev)
    fl = head.from_logits(sq, cls); tl = head.to_logits(sq, fr, cls)
    if base is not None:
        with torch.no_grad():
            bfl = base.from_logits(sq, cls); btl = base.to_logits(sq, fr, cls)
        fl = fl + bfl; tl = tl + btl
    return fl, tl


def train_head(D, rows, dev, epochs=2, wd=0.0, base=None, seed=0):
    head = BandHead(384, 512, use_cls=True).to(dev)
    if base is not None: _zero_last(head.from_head); _zero_last(head.to_head)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=wd)
    sub = {k: v[rows] for k, v in D.items()}; head.train()
    for ep in range(epochs):
        for b in _b(len(rows), True, seed*10+ep):
            fl, tl = _logits(head, sub, b, dev, base)
            loss = masked_square_ce(fl, sub["fr"][b].to(dev), sub["fm"][b].to(dev)) + \
                   masked_square_ce(tl, sub["to"][b].to(dev), sub["tm"][b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval(); return head


@torch.no_grad()
def per_pos_ce(head, D, rows, dev, base):
    """(from+to) CE per row (numpy) for head+base over rows."""
    sub = {k: v[rows] for k, v in D.items()}; out = np.empty(len(rows), np.float32)
    for b in _b(len(rows)):
        fl, tl = _logits(head, sub, b, dev, base)
        fl = fl.masked_fill(~sub["fm"][b].to(dev), _NEG); tl = tl.masked_fill(~sub["tm"][b].to(dev), _NEG)
        ce = (torch.nn.functional.cross_entropy(fl, sub["fr"][b].to(dev), reduction="none")
              + torch.nn.functional.cross_entropy(tl, sub["to"][b].to(dev), reduction="none"))
        out[b.numpy()] = ce.cpu().numpy()
    return out


@torch.no_grad()
def per_pos_mm(head, D, rows, dev, base):
    sub = {k: v[rows] for k, v in D.items()}; out = np.zeros(len(rows), bool)
    for b in _b(len(rows)):
        fl, tl = _logits(head, sub, b, dev, base)
        pf = fl.masked_fill(~sub["fm"][b].to(dev), _NEG).argmax(-1)
        pt = tl.masked_fill(~sub["tm"][b].to(dev), _NEG).argmax(-1)
        out[b.numpy()] = ((pf == sub["fr"][b].to(dev)) & (pt == sub["to"][b].to(dev))).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/mnt/eloquence_bulk/databases/style_feat_cache.h5")
    ap.add_argument("--seed-assign", default="/workspaces/eloquent-encoding/style_clustering_out/metric_assignment.npz")
    ap.add_argument("--test-per-unit", type=int, default=40)
    ap.add_argument("--baseline-train", type=int, default=120000)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); dev = a.device; rng = np.random.default_rng(a.seed)

    A = np.load(a.seed_assign, allow_pickle=True)
    seed_cl = {int(u): int(c) for u, c, nv in zip(A["unit_id"], A["cluster"], A["is_novel"]) if not nv}
    K = max(seed_cl.values()) + 1
    f = h5py.File(a.cache, "r"); uid = f["unit_id"][:]; split = f["split"][:]
    test_by = {}; train_pool = []
    for r in range(uid.shape[0]):
        u = int(uid[r])
        if u not in seed_cl: continue
        (test_by.setdefault(u, []).append(r) if split[r] == 1 else train_pool.append(r))
    ev_rows = []; ev_u = []
    for u, rs in test_by.items():
        for r in rs[:a.test_per_unit]: ev_rows.append(r); ev_u.append(u)
    ev_rows = np.array(ev_rows); ev_u = np.array(ev_u)
    tr = rng.choice(np.array(train_pool), min(a.baseline_train, len(train_pool)), replace=False)
    tr_u = np.array([int(uid[r]) for r in tr])
    eo = np.argsort(ev_rows); ev_rows, ev_u = ev_rows[eo], ev_u[eo]
    to = np.argsort(tr); tr, tr_u = tr[to], tr_u[to]
    units = sorted(seed_cl); u2i = {u: i for i, u in enumerate(units)}; nU = len(units)
    tr_ui = np.array([u2i[u] for u in tr_u]); ev_ui = np.array([u2i[u] for u in ev_u])
    print(f"eval {len(ev_rows):,} / train {len(tr):,} | {nU} units, K={K}", flush=True)

    E = _read(f, ev_rows); T = _read(f, tr)
    H = train_head(T, np.arange(len(tr)), dev, epochs=2, wd=0.0, seed=0)
    assign = np.array([seed_cl[u] for u in units])  # per-unit-index cluster

    def heldout(assign):
        # weighted (H+r_assigned) move% and CE on test, per current residuals
        mm = np.empty(len(ev_rows)); ce = np.empty(len(ev_rows))
        for k in range(K):
            rows_k = np.where(assign[ev_ui] == k)[0]
            if len(rows_k) == 0: continue
            mm[rows_k] = per_pos_mm(rk[k], E, rows_k, dev, H)
            ce[rows_k] = per_pos_ce(rk[k], E, rows_k, dev, H)
        mmH = per_pos_mm(H, E, np.arange(len(ev_rows)), dev, None)
        ceH = per_pos_ce(H, E, np.arange(len(ev_rows)), dev, None)
        return 100*mm.mean(), 100*mmH.mean(), ce.mean(), ceH.mean()

    for it in range(a.iters):
        rk = {}
        for k in range(K):
            rows_k = np.where(assign[tr_ui] == k)[0]
            rk[k] = train_head(T, rows_k, dev, epochs=2, wd=a.wd, base=H, seed=k+1) if len(rows_k) >= 100 else None
        # held-out with current assignment
        mmHr, mmH, ceHr, ceH = heldout(assign)
        # M-step: reassign by per-unit train CE under each residual
        ce_uk = np.full((nU, K), np.inf)
        for k in range(K):
            if rk[k] is None: continue
            cep = per_pos_ce(rk[k], T, np.arange(len(tr)), dev, H)
            s = np.bincount(tr_ui, weights=cep, minlength=nU); c = np.bincount(tr_ui, minlength=nU)
            ce_uk[:, k] = np.where(c > 0, s / np.maximum(c, 1), np.inf)
        new = ce_uk.argmin(1)
        changed = (new != assign).mean()
        sizes = [int((new == k).sum()) for k in range(K)]
        print(f"[iter {it}] heldout move% H={mmH:.2f} H+r={mmHr:.2f} ({mmHr-mmH:+.3f}) | "
              f"CE {ceH:.4f}->{ceHr:.4f} ({ceHr-ceH:+.4f}) | reassigned {100*changed:.1f}% | sizes {sizes}", flush=True)
        assign = new


if __name__ == "__main__":
    main()
