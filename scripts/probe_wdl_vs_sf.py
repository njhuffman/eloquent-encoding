#!/usr/bin/env python3
"""Probe: how well does the model's WDL value head track Stockfish on the validation set?

Computes the model's STM-relative escore (P(win)+0.5 P(draw)) for every validation position, joins
the Stockfish sidecar (depth-8 cp/wdl, static NNUE cp, mate), and reports correlations, a linear
probe R^2 (model WDL -> SF escore), mate discrimination (AUC), and a calibration table. Pure numpy
metrics (no scipy/sklearn).
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import h5py
from style_policy.model import BasePolicy
from style_policy.model_spec import elo_to_bucket


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return pearson(ra.astype(float), rb.astype(float))


def auc(scores, labels):  # rank-based Mann-Whitney AUC; labels bool
    pos = labels.astype(bool); neg = ~pos
    np_, nn_ = pos.sum(), neg.sum()
    if np_ == 0 or nn_ == 0:
        return float("nan")
    order = np.argsort(scores); ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[pos].sum() - np_ * (np_ + 1) / 2) / (np_ * nn_))


def _split(n, seed):
    idx = np.random.default_rng(seed).permutation(n); k = int(0.8 * n)
    return idx[:k], idx[k:]


def _r2(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum(); ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def linear_probe_r2(X, y, seed=0):  # OLS fit on 80%, report test R^2
    tr, te = _split(len(y), seed)
    Xa = np.hstack([X, np.ones((len(y), 1))])
    coef, *_ = np.linalg.lstsq(Xa[tr], y[tr], rcond=None)
    return _r2(y[te], Xa[te] @ coef)


def ridge_probe_r2(X, y, lam=10.0, seed=0):  # standardized ridge on 80%, test R^2 (bias unregularized)
    tr, te = _split(len(y), seed)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xs = (X - mu) / sd
    Xtr = np.hstack([Xs[tr], np.ones((len(tr), 1))]); Xte = np.hstack([Xs[te], np.ones((len(te), 1))])
    d = Xtr.shape[1]; A = Xtr.T @ Xtr + lam * np.eye(d); A[-1, -1] -= lam
    w = np.linalg.solve(A, Xtr.T @ y[tr])
    return _r2(y[te], Xte @ w)


def mlp_probe_r2(X, y, device, hidden=128, epochs=500, lr=1e-3, wd=1e-4, seed=0):
    """Small MLP (in->hidden->1, GELU) on frozen features; standardized x/y, full-batch Adam; test R^2."""
    torch.manual_seed(seed)
    tr, te = _split(len(y), seed)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    ym, ys = y[tr].mean(), y[tr].std() + 1e-6
    Xtr = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32, device=device)
    Xte = torch.tensor((X[te] - mu) / sd, dtype=torch.float32, device=device)
    ytr = torch.tensor((y[tr] - ym) / ys, dtype=torch.float32, device=device).unsqueeze(1)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(),
                              torch.nn.Linear(hidden, 1)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad(); torch.nn.functional.mse_loss(net(Xtr), ytr).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(Xte).squeeze(1).cpu().numpy() * ys + ym
    return _r2(y[te], pred)


@torch.no_grad()
def model_features(checkpoint, packed, elo, device, batch=1024):
    """Return (probs (N,3) [loss,draw,win], cls (N,d_model)) — the WDL output and the encoding it reads."""
    ck = torch.load(checkpoint, map_location=device)
    m = BasePolicy.from_config(ck["architecture"]); m.load_state_dict(ck["model"]); m.to(device).eval()
    n_elo = int(ck["architecture"]["n_elo_buckets"]); d = int(ck["architecture"]["d_model"])
    probs = np.empty((len(packed), 3), dtype=np.float64)
    cls = np.empty((len(packed), d), dtype=np.float64)
    for i in range(0, len(packed), batch):
        pk = torch.from_numpy(packed[i:i + batch]).to(device)
        idx = elo_to_bucket(torch.from_numpy(elo[i:i + batch].astype(np.int64)), n_elo).to(device)
        c = m.encode(pk)[0]                                  # (B, d_model) CLS embedding
        probs[i:i + batch] = torch.softmax(m.value_head(c, elo_idx=idx), dim=-1).cpu().numpy()
        cls[i:i + batch] = c.cpu().numpy()
    return probs, cls


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-h5", default="/mnt/eloquence_bulk/databases/wdl_validation_1M.h5")
    ap.add_argument("--sidecar", default="/mnt/eloquence_bulk/databases/wdl_validation_1M.sf_eval.h5")
    ap.add_argument("--checkpoint", default="style_policy_checkpoints/wdl_16M/wdl_16M_stage_1.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    with h5py.File(args.sidecar, "r") as s:
        rows = s["row_index"][:]
        sf_cp = s["sf_cp"][:].astype(np.float64)
        sf_static = s["sf_static_cp"][:].astype(np.float64)
        sf_mate = s["sf_mate"][:].astype(np.int64)
        sf_wdl = s["sf_wdl"][:].astype(np.float64)  # [loss,draw,win] permille
    with h5py.File(args.val_h5, "r") as f:
        packed = f["packed_pre"][:][rows].astype(np.uint8)
        elo = f["elo_to_move"][:][rows]

    probs, cls = model_features(args.checkpoint, packed, elo, args.device)
    m_es = probs[:, 2] + 0.5 * probs[:, 1]                     # model escore, STM
    sf_es = (sf_wdl[:, 2] + 0.5 * sf_wdl[:, 1]) / 1000.0       # SF depth-8 escore, STM
    has_wdl = sf_wdl.sum(axis=1) > 0
    ok_static = sf_static != -32768

    n = len(m_es)
    print(f"n={n}  (wdl present {has_wdl.mean()*100:.1f}%, static defined {ok_static.mean()*100:.1f}%, "
          f"forced-mate seen {np.mean(sf_mate!=0)*100:.1f}%)\n")

    print("model escore vs Stockfish:")
    print(f"  vs SF depth-8 escore : pearson {pearson(m_es[has_wdl], sf_es[has_wdl]):+.3f}  "
          f"spearman {spearman(m_es[has_wdl], sf_es[has_wdl]):+.3f}  "
          f"MAE {np.abs(m_es[has_wdl]-sf_es[has_wdl]).mean():.3f}")
    print(f"  vs SF depth-8 cp     : spearman {spearman(m_es, sf_cp):+.3f}")
    print(f"  vs SF static NNUE cp : spearman {spearman(m_es[ok_static], sf_static[ok_static]):+.3f}\n")

    print("probe test R^2:  3-prob (WDL output) | cls-256 linear | cls-256 MLP(128):")
    for name, m, tgt in [
        ("SF depth-8 escore", has_wdl, sf_es),
        ("SF depth-8 cp(±1500)", np.ones(n, bool), np.clip(sf_cp, -1500, 1500)),
        ("SF static cp(±1500)", ok_static, np.clip(sf_static, -1500, 1500)),
    ]:
        r3 = linear_probe_r2(probs[m], tgt[m])
        rl = ridge_probe_r2(cls[m], tgt[m])
        rm = mlp_probe_r2(cls[m], tgt[m], args.device)
        print(f"  {name:22s}: 3-prob {r3:.3f}   cls-lin {rl:.3f}   cls-mlp {rm:.3f}")
    print()

    mate = sf_mate != 0
    if mate.sum():
        win_mate = sf_mate > 0
        print("forced-mate positions (SF):")
        print(f"  mean model escore  STM-mating(+): {m_es[mate & win_mate].mean():.3f}   "
              f"STM-mated(−): {m_es[mate & ~win_mate].mean():.3f}")
        print(f"  AUC model escore distinguishes mate+ vs mate− : {auc(m_es[mate], win_mate[mate]):.3f}\n")

    print("calibration  (model escore decile -> mean SF depth-8 escore):")
    order = np.argsort(m_es[has_wdl]); me = m_es[has_wdl][order]; se = sf_es[has_wdl][order]
    for d in range(10):
        lo, hi = d * len(me) // 10, (d + 1) * len(me) // 10
        print(f"  decile {d}: model {me[lo:hi].mean():.3f}  ->  SF {se[lo:hi].mean():.3f}  (n={hi-lo})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
