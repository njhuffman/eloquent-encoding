"""Rigorous concept probe: closed-form ridge (deterministic given split) + n=20k + K-seed mean±std.
Fixes the huge SGD-probe variance of concept_probe.py (std ~0.09 / range ~0.3 at n=8000 single-seed).
Global concepts only (per-square maps are saturated/noise-free). Loads a FlatMultiTaskPolicy ckpt.

Usage: python scripts/concept_probe_v2.py --ckpt <resume.pt> --model flat_multitask_128M --device cuda --n 20000 --k 5
"""
from __future__ import annotations
import argparse, sys, os, numpy as np, torch, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import concept_probe as cp                                   # reuse CONCEPTS, labels, encode, board_planes
from style_policy.board_encode import packed_to_board
from style_policy.flat_policy import FlatMultiTaskPolicy
from style_policy.model_spec import load_spec


def ridge_probe(X, y, kind, seed, alpha=10.0):
    """Closed-form ridge probe on standardized features -> held-out R2 (reg) or accuracy (bin).
    Deterministic given the split, so K seeds measure ONLY train/test-split variance."""
    g = torch.Generator().manual_seed(seed)
    n = len(X); perm = torch.randperm(n, generator=g); ntr = int(0.8 * n)
    tr, te = perm[:ntr], perm[ntr:]
    Xtr, Xte = X[tr], X[te]
    mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    Xtr = torch.cat([Xtr, torch.ones(len(Xtr), 1)], 1); Xte = torch.cat([Xte, torch.ones(len(Xte), 1)], 1)
    d = Xtr.shape[1]
    ytr = y[tr].float(); yte = y[te].float()
    w = torch.linalg.solve(Xtr.T @ Xtr + alpha * torch.eye(d), Xtr.T @ ytr)
    pred = Xte @ w
    if kind == "reg":
        ss_res = ((pred - yte) ** 2).sum(); ss_tot = ((yte - yte.mean()) ** 2).sum().clamp_min(1e-6)
        return float(1 - ss_res / ss_tot)
    return float(((pred > 0.5).float() == yte).float().mean())    # least-squares classification


def load_model(ckpt, policy, model_name, dev, random_init=False):
    """policy='flat' -> FlatMultiTaskPolicy (arch from config, resume.pt has no 'architecture').
       policy='multiband' -> MultiBandPolicy (arch from the checkpoint itself)."""
    if policy == "multiband":
        from style_policy.multiband_policy import MultiBandPolicy
        ck = torch.load(ckpt, map_location=dev, weights_only=False)
        m = MultiBandPolicy.from_config(ck["architecture"]).to(dev).eval()
        if not random_init:
            sd = {k.replace("encoder._orig_mod.", "encoder.", 1): v for k, v in ck["model"].items()}
            m.load_state_dict(sd, strict=False)
        return m
    arch = load_spec(model_name)["architecture"]
    m = FlatMultiTaskPolicy.from_config(arch).to(dev).eval()
    if not random_init:
        st = torch.load(ckpt, map_location=dev, weights_only=False)["model"]
        st = {k.replace("encoder._orig_mod.", "encoder.", 1): v for k, v in st.items()}
        m.load_state_dict(st, strict=False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="flat_multitask_128M")
    ap.add_argument("--val", default="/mnt/eloquence_bulk/databases/wdl_validation_2025_05.h5")
    ap.add_argument("--n", type=int, default=20000); ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--tag", default="")
    ap.add_argument("--policy", default="flat", choices=["flat", "multiband"])
    a = ap.parse_args(); dev = a.device if torch.cuda.is_available() else "cpu"

    f = h5py.File(a.val, "r"); tot = f["packed_pre"].shape[0]
    idx = np.sort(np.random.default_rng(0).choice(tot, a.n, replace=False))
    packed = f["packed_pre"][idx]
    print(f"reconstructing {a.n:,} boards + labels ...", flush=True)
    boards = [packed_to_board(p.astype(np.uint8)) for p in packed]
    glob_concepts = [(nm, k, t, fn) for (nm, k, t, fn, _) in cp.CONCEPTS if k == "global"]
    labels = {nm: torch.from_numpy(np.array([fn(b) for b in boards])) for (nm, k, t, fn) in glob_concepts}
    raw = torch.from_numpy(np.stack([cp.board_planes(b) for b in boards])).float()

    print("encoding trained + random (GPU) ...", flush=True)
    mt = load_model(a.ckpt, a.policy, a.model, dev); cT, sT = cp.encode(mt, packed, dev); del mt
    mr = load_model(a.ckpt, a.policy, a.model, dev, random_init=True); cR, sR = cp.encode(mr, packed, dev); del mr
    if dev == "cuda": torch.cuda.empty_cache()
    globT = torch.cat([cT, sT.mean(1)], 1); globR = torch.cat([cR, sR.mean(1)], 1)

    print(f"\n===== RIGOROUS CONCEPT PROBE (ridge, n={a.n:,}, K={a.k} seeds){' '+a.tag if a.tag else ''} =====")
    print(f"{'concept':<15}{'trained (mean±std)':>22}{'random':>16}{'raw-inp':>16}")
    for (nm, kind, tgt, fn) in glob_concepts:
        y = labels[nm]
        def band(X):
            s = np.array([ridge_probe(X, y, tgt, seed) for seed in range(a.k)])
            return s.mean(), s.std()
        (tm, ts), (rm, rs), (wm, ws) = band(globT), band(globR), band(raw)
        print(f"{nm:<15}{tm:>13.3f} ±{ts:.3f}{rm:>11.3f} ±{rs:.3f}{wm:>11.3f} ±{ws:.3f}", flush=True)


if __name__ == "__main__":
    main()
