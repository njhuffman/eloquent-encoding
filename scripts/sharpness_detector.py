"""Detector step for the GAIL-revisit idea: can a discriminator separate HUMAN game states from the
model's SELF-PLAY states, and how much does SHARPNESS carry the signal vs encoder features alone
(the old GAIL basis)? If sharpness features give high AUC and add over encoder-only, sharpness is a
clean, learnable, temperature-orthogonal axis -> a usable GAIL reward.

Feature sets: ENC = [cls ++ mean-square] encoder features; SHARP = [blunder_density, eval_spread,
current_eval, n_legal]; BOTH. Ridge-logistic probe, held-out ROC-AUC (K seeds)."""
from __future__ import annotations
import argparse, sys, os, numpy as np, torch, chess, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from style_policy.flat_policy import FlatMultiTaskPolicy
from style_policy.board_encode import packed_to_board, board_to_packed
from style_policy.model_spec import elo_to_bucket
from style_policy import move_index
import position_sharpness as ps                              # reuse Model + sharpness() + selfplay_positions + human_positions

DEV = "cuda"; NEG = -1e9


@torch.no_grad()
def enc_feats(mo, boards):
    out = []
    for i in range(0, len(boards), 512):
        pk = torch.from_numpy(np.stack([board_to_packed(b) for b in boards[i:i+512]]).astype(np.int64)).to(DEV)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            cls, sq = mo.m.encode(pk, hist=None)
        out.append(torch.cat([cls.float(), sq.float().mean(1)], 1).cpu().numpy())
    return np.concatenate(out)


def sharp_feats(mo, boards):
    dens, spread = ps.sharpness(mo, boards)
    V = mo.value(np.stack([board_to_packed(b) for b in boards]).astype(np.int64))
    nl = np.array([b.legal_moves.count() for b in boards], dtype=np.float32)
    return np.stack([dens, spread, V, np.log(nl + 1)], 1)


def auc(X, y, k=4):
    n = len(y); aucs = []
    for s in range(k):
        rng = np.random.default_rng(s); perm = rng.permutation(n); tr, te = perm[:int(.8*n)], perm[int(.8*n):]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        clf = LogisticRegression(max_iter=2000, C=1.0).fit((X[tr]-mu)/sd, y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba((X[te]-mu)/sd)[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/eloquence_bulk/databases/wdl_history_128M.h5")
    ap.add_argument("--band", type=int, default=1500); ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--games", type=int, default=60); ap.add_argument("--w", type=int, default=60)
    a = ap.parse_args()
    mo = ps.Model(); f = h5py.File(a.data, "r"); print("loading elo ...", flush=True); elo = f["elo_to_move"][:]
    hb = ps.human_positions(f, elo, a.band, a.w, a.n)
    sb = ps.selfplay_positions(mo, a.band, a.games, 160, 1.0, a.n)
    n = min(len(hb), len(sb)); hb, sb = hb[:n], sb[:n]
    print(f"band {a.band}: {n} human + {n} self-play states", flush=True)
    Xe = np.concatenate([enc_feats(mo, hb), enc_feats(mo, sb)])
    Xs = np.concatenate([sharp_feats(mo, hb), sharp_feats(mo, sb)])
    y = np.concatenate([np.ones(n), np.zeros(n)])            # human=1, model=0
    print(f"\n===== HUMAN vs MODEL-SELF-PLAY detector AUC (band {a.band}, n={2*n}) =====")
    for name, X in [("encoder feats (old GAIL basis)", Xe), ("sharpness feats", Xs),
                    ("BOTH", np.concatenate([Xe, Xs], 1))]:
        m, s = auc(X, y); print(f"  {name:<34} AUC {m:.3f} ±{s:.3f}", flush=True)
    # which sharpness feature carries it
    names = ["blunder_density", "eval_spread", "current_eval", "log_nlegal"]
    for j, nm in enumerate(names):
        m, _ = auc(Xs[:, j:j+1], y); print(f"    [{nm:<16}] alone AUC {m:.3f}", flush=True)


if __name__ == "__main__":
    main()
