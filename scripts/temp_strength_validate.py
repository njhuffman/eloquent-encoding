"""No-training validation of the "GAIL for precise-level play" idea. Question: does TEMPERATURE
(the current strength knob) reproduce a band's PLAY, or only its average strength? If, at the
temperature T* that matches band-B strength, the bot's states are still distinguishable from real
band-B human states, then temperature is crude and GAIL (occupancy-matching) has a real target.

For temps in a sweep, seed the bot from real human band-B openings, roll forward, then measure:
  (a) apparent strength = strength-classifier predicted elo on the bot's states
  (b) AUC = discriminator(bot states vs real band-B human continuation states)
Plus band-separability reference (B vs a higher band) + human A/B floor. 128M big model, GPU.
"""
from __future__ import annotations
import argparse, sys, os, numpy as np, torch, h5py, chess
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selfplay_drift import load, feats, disc_auc, _hist_tensors, _cap
from selfplay_drift_seeded import collect_seeds
from style_policy.board_encode import board_to_packed

BANDS = list(range(1000, 2200, 100))
def bcenter(b): return 1000 + 100 * b + 50


def train_strength_clf(model, data_h5, per_band, dev):
    """12-way band classifier on frozen features (cls+meansq); returns predict_elo(packed)->mean elo."""
    f = h5py.File(data_h5, "r"); elo = f["elo_to_move"][:]; band = np.clip((elo // 100) * 100, 1000, 2100)
    rng = np.random.default_rng(0)
    sel = np.sort(np.concatenate([rng.choice(np.where(band == b)[0], min(per_band, int((band == b).sum())), replace=False) for b in BANDS]))
    y = torch.tensor([min(11, max(0, int(elo[i]) // 100 - 10)) for i in sel])
    hist = (torch.from_numpy(f["hist_from"][sel][:, :2].astype(np.int64)),
            torch.from_numpy(f["hist_to"][sel][:, :2].astype(np.int64)),
            torch.from_numpy(f["hist_cap"][sel][:, :2].astype(np.int64)))
    # encode WITH history to match training-time distribution for the classifier's inputs
    X = _feats_hist(model, f["packed_pre"][sel], hist, dev)
    clf = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, 12)).to(dev)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(30):
        p = torch.randperm(len(X))
        for i in range(0, len(X), 4096):
            b = p[i:i+4096]
            loss = torch.nn.functional.cross_entropy(clf(X[b].to(dev)), y[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
    clf.eval(); centers = torch.tensor([bcenter(b) for b in range(12)], dtype=torch.float32)

    @torch.no_grad()
    def predict_elo(packed):
        F = feats(model, packed, dev)  # hist=None (state-only), consistent for bot states
        probs = torch.softmax(clf(F.to(dev)), 1).cpu()
        return float((probs @ centers).mean())
    return predict_elo


@torch.no_grad()
def _feats_hist(model, packed, hist, dev, bs=256):
    C, S = [], []
    for i in range(0, len(packed), bs):
        pk = torch.from_numpy(np.asarray(packed[i:i+bs]).astype(np.int64)).to(dev)
        h = tuple(x[i:i+bs].to(dev) for x in hist)
        c, s = model.encode(pk, hist=h); C.append(c.float().cpu()); S.append(s.float().mean(1).cpu())
    return torch.cat([torch.cat(C), torch.cat(S)], 1)


@torch.no_grad()
def rollout_temp(model, head, n_ply, seeds, K, temp, dev, gseed=0):
    g = torch.Generator(device=dev).manual_seed(gseed)
    boards = [s[0].copy() for s in seeds]; recents = [deque(s[1], maxlen=max(n_ply, 1)) for s in seeds]
    for step in range(K):
        act = [i for i in range(len(boards)) if not boards[i].is_game_over()]
        if not act: break
        packed = np.stack([board_to_packed(boards[i]) for i in act])
        hist = _hist_tensors([recents[i] for i in act], n_ply, dev) if n_ply else None
        cls, sq = model.encode(torch.from_numpy(packed.astype(np.int64)).to(dev), hist=hist)
        for k, i in enumerate(act):
            b = boards[i]; by = {}
            for m in b.legal_moves:
                key = (m.from_square, m.to_square)
                if key not in by or m.promotion == chess.QUEEN: by[key] = m
            froms = sorted({f for f, _ in by})
            fl = head.from_logits(sq[k:k+1], cls[k:k+1])[0][froms]
            fi = torch.multinomial(torch.softmax(fl / temp, -1), 1, generator=g).item()
            f = froms[fi]; tos = [t for (ff, t) in by if ff == f]
            tl = head.to_logits(sq[k:k+1], torch.tensor([f], device=dev), cls[k:k+1])[0][tos]
            ti = torch.multinomial(torch.softmax(tl / temp, -1), 1, generator=g).item()
            mv = by[(f, tos[ti])]; recents[i].append((f, tos[ti], _cap(b, mv))); b.push(mv)
    return np.array([board_to_packed(b) for b in boards])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="style_policy_checkpoints/multiband_history_128M_big/multiband_history_128M_big.pt")
    ap.add_argument("--pgn", default="/mnt/eloquence_bulk/databases/lichess_db_standard_rated_2025-05_tc_600_0.pgn.zst")
    ap.add_argument("--data", default="/mnt/eloquence_bulk/databases/wdl_history_128M.h5")
    ap.add_argument("--band", type=int, default=1500); ap.add_argument("--ref-band", type=int, default=1900)
    ap.add_argument("--seed-ply", type=int, default=10); ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--n-seeds", type=int, default=1200)
    ap.add_argument("--temps", default="0.1,0.3,0.5,0.7,1.0"); ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); dev = a.device; temps = [float(x) for x in a.temps.split(",")]
    model, n_ply = load(a.ckpt, dev)
    head = model.heads[int(model.head_index(torch.tensor([a.band])).item())]
    print("training strength classifier ...", flush=True)
    predict_elo = train_strength_clf(model, a.data, 6000, dev)
    print(f"collecting {a.n_seeds} human seeds (band {a.band}, ply {a.seed_ply}) ...", flush=True)
    seeds = collect_seeds(a.pgn, a.band, a.seed_ply, [a.horizon], a.n_seeds, n_ply)
    print(f"  {len(seeds)} seeds", flush=True)

    human = np.stack([board_to_packed(s[2][a.horizon]) for s in seeds]); Fh = feats(model, human, dev)
    print(f"  human continuation apparent-elo = {predict_elo(human):.0f} (target {a.band})", flush=True)

    print(f"\n===== TEMPERATURE vs BAND-{a.band} PLAY (seed ply {a.seed_ply}, +{a.horizon} plies) =====")
    print(f"  {'temp':>5} {'apparent-elo':>13} {'AUC vs '+str(a.band)+'-human':>18}")
    for t in temps:
        bot = rollout_temp(model, head, n_ply, seeds, a.horizon, t, dev)
        Fb = feats(model, bot, dev)
        print(f"  {t:>5.2f} {predict_elo(bot):>13.0f} {disc_auc(Fh, Fb, dev):>18.3f}", flush=True)

    nh = len(Fh) // 2
    print(f"\n  floor (human {a.band} A/B): AUC {disc_auc(Fh[:nh], Fh[nh:], dev):.3f}")
    # band separability reference
    f = h5py.File(a.data, "r"); elo = f["elo_to_move"][:]
    ri = np.where((elo >= a.ref_band) & (elo < a.ref_band + 100))[0]
    rng = np.random.default_rng(1); ri = np.sort(rng.choice(ri, min(len(human), len(ri)), replace=False))
    Fr = feats(model, f["packed_pre"][ri], dev)
    print(f"  band separability ({a.band} vs {a.ref_band} humans): AUC {disc_auc(Fh, Fr, dev):.3f}")
    print("  read: find temp where apparent-elo~=target; if its AUC-vs-human >> floor => temperature")
    print("        gives right strength but wrong play => GAIL opportunity. If ~floor => temp suffices.")


if __name__ == "__main__":
    main()
