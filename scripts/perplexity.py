#!/usr/bin/env python3
"""Per-move perplexity of the human move for our MultiBandPolicy or Maia2, matched:
P(move|legal) = P(from)*P(to|from) (our factored heads -> a proper distribution over legal moves)
vs Maia's move distribution renormalized over legal. Reports raw (T=1) and temperature-CALIBRATED
perplexity (best T on this set) per band + overall, so the label-smoothing/calibration confound is
removed. Lower = better."""
from __future__ import annotations
import argparse
import numpy as np
import torch
import h5py
from style_policy.legal_mask import u64_to_mask

_TEMPS = [0.5, 0.6, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]
_NEG = float("-inf")


def _report(nll_by_T, band_of, name):
    # nll_by_T: (T, n) array of per-move NLL
    means = nll_by_T.mean(axis=1)
    best = int(means.argmin())
    print(f"\n{name}: perplexity  raw(T=1)={np.exp(means[_TEMPS.index(1.0)]):.3f}   "
          f"calibrated(T={_TEMPS[best]})={np.exp(means[best]):.3f}")
    print(f"  {'band':>9} {'ppl@T=1':>8} {'ppl@bestT':>9}")
    for b in range(12):
        m = band_of == b
        if m.any():
            r = np.exp(nll_by_T[_TEMPS.index(1.0), m].mean())
            c = np.exp(nll_by_T[best, m].mean())
            print(f"  {1000+100*b:>4}-{1099+100*b:<4} {r:>8.3f} {c:>9.3f}")


@torch.no_grad()
def run_multiband(a):
    from style_policy.multiband_policy import MultiBandPolicy
    ck = torch.load(a.ckpt, map_location=a.device)
    model = MultiBandPolicy.from_config(ck["architecture"]); model.load_state_dict(ck["model"]); model.to(a.device).eval()
    nply = int(ck["architecture"].get("n_history_ply", 0)) if ck["architecture"].get("use_last_move") else 0
    with h5py.File(a.val_h5, "r") as f:
        n = min(a.n, int(f["packed_pre"].shape[0]))
        packed = torch.from_numpy(f["packed_pre"][:n].astype(np.uint8))
        fs = torch.from_numpy(f["from_sq"][:n].astype(np.int64)); ts = torch.from_numpy(f["to_sq"][:n].astype(np.int64))
        fmu = torch.from_numpy(np.array(f["from_legal_u64"][:n], dtype=np.uint64)).to(torch.int64)
        tmu = torch.from_numpy(np.array(f["to_legal_u64"][:n], dtype=np.uint64)).to(torch.int64)
        elo = torch.from_numpy(f["elo_to_move"][:n].astype(np.int64))
        if nply and a.k and "hist_from" in f:
            hf = torch.from_numpy(f["hist_from"][:n].astype(np.int64)); ht = torch.from_numpy(f["hist_to"][:n].astype(np.int64)); hc = torch.from_numpy(f["hist_cap"][:n].astype(np.int64))
            if a.k < hf.shape[1]:
                hf[:, a.k:] = -1; ht[:, a.k:] = -1; hc[:, a.k:] = 0
        else:
            hf = ht = hc = None
    band_of = model.head_index(elo).numpy()
    nll = np.zeros((len(_TEMPS), n), dtype=np.float64)
    for i in range(0, n, a.batch):
        sl = slice(i, min(i + a.batch, n)); rows = torch.arange(sl.stop - sl.start)
        hist = (hf[sl].to(a.device), ht[sl].to(a.device), hc[sl].to(a.device)) if hf is not None else None
        cls, sq = model.encode(packed[sl], hist=hist)
        fm = u64_to_mask(fmu[sl].to(a.device)); tm = u64_to_mask(tmu[sl].to(a.device))
        f_t = fs[sl].to(a.device); t_t = ts[sl].to(a.device)
        hidx = torch.from_numpy(band_of[sl])
        fl = torch.empty(len(rows), 64, device=a.device); tl = torch.empty(len(rows), 64, device=a.device)
        for g in range(model.n_bands):
            m = hidx == g
            if m.any():
                fl[m] = model.heads[g].from_logits(sq[m], cls[m]).float()
                tl[m] = model.heads[g].to_logits(sq[m], f_t[m], cls[m]).float()
        fl = fl.masked_fill(~fm, _NEG); tl = tl.masked_fill(~tm, _NEG)
        for ti, T in enumerate(_TEMPS):
            flp = torch.log_softmax(fl / T, -1)[rows, f_t]
            tlp = torch.log_softmax(tl / T, -1)[rows, t_t]
            nll[ti, sl] = (-(flp + tlp)).cpu().numpy()
    _report(nll, band_of, f"MultiBand K={a.k}")


def run_maia(a):
    import chess
    from style_policy.board_encode import packed_to_board
    from style_policy.maia2_bot import load_maia2
    from maia2 import inference
    maia, prep = load_maia2("rapid", device=("gpu" if str(a.device).startswith("cuda") else "cpu"))
    with h5py.File(a.val_h5, "r") as f:
        n = min(a.n, int(f["packed_pre"].shape[0]))
        packed = f["packed_pre"][:n]; fs = f["from_sq"][:n].astype(int); ts = f["to_sq"][:n].astype(int); elo = f["elo_to_move"][:n].astype(int)
    band_of = np.clip((np.clip(elo, 1000, 2199) - 1000) // 100, 0, 11)
    nll = np.zeros((len(_TEMPS), n), dtype=np.float64); keep = np.ones(n, bool)
    for i in range(n):
        b = packed_to_board(np.asarray(packed[i], np.uint8))
        if b.is_game_over(): keep[i] = False; continue
        se = int(min(1900, max(1000, elo[i])))
        mp, _ = inference.inference_each(maia, prep, b.fen(), se, se)
        legal = [m.uci() for m in b.legal_moves]
        p = np.array([max(mp.get(u, 0.0), 1e-12) for u in legal])
        true_uci = chess.Move(int(fs[i]), int(ts[i])).uci()
        ti_true = legal.index(true_uci) if true_uci in legal else (legal.index(chess.Move(int(fs[i]), int(ts[i]), promotion=chess.QUEEN).uci()) if chess.Move(int(fs[i]), int(ts[i]), promotion=chess.QUEEN).uci() in legal else None)
        if ti_true is None: keep[i] = False; continue
        for k, T in enumerate(_TEMPS):
            pT = p ** (1.0 / T); pT = pT / pT.sum()
            nll[k, i] = -np.log(pT[ti_true])
        if i and i % 2000 == 0: print(f"  ...{i}/{n}", flush=True)
    _report(nll[:, keep], band_of[keep], "Maia2")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["multiband", "maia"], required=True)
    ap.add_argument("--ckpt", default="style_policy_checkpoints/multiband_history_128M_big/multiband_history_128M_big.pt")
    ap.add_argument("--val-h5", default="/mnt/eloquence_bulk/databases/wdl_validation_2025_05.h5")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    (run_multiband if a.model == "multiband" else run_maia)(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
