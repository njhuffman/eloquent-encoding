"""Part 1: state-plausibility probes on the FROZEN big encoder (CLS token only).
(A) ELO-band classifier   : P(elo band | position)          -> does the encoder encode STRENGTH?
(B) Naturalness discriminator: is an afterstate reached by a HUMAN move or a RANDOM legal move?
Both = tiny MLP heads on the frozen encoder's CLS (position-global judgments). CLS features are
384-d -> the whole cache is tiny, fits in RAM, no IO wall.
"""
from __future__ import annotations
import argparse, numpy as np, torch, h5py, chess, random
from style_policy.multiband_policy import MultiBandPolicy
from style_policy.board_encode import board_to_packed, packed_to_board

BANDS = list(range(1000, 2200, 100))  # 12 bands


def load_encoder(ckpt, dev):
    ck = torch.load(ckpt, map_location=dev)
    arch = ck["architecture"]
    m = MultiBandPolicy.from_config(arch).to(dev).eval(); m.load_state_dict(ck["model"])
    n_ply = int(arch.get("n_history_ply", 0)) if arch.get("use_last_move") else 0
    return m, n_ply


@torch.no_grad()
def encode_feats(model, packed, hist, dev, bs=256):
    """Return (cls (N,384), meansq (N,384)) — CLS token and mean-pooled square tokens."""
    C, S = [], []
    for i in range(0, packed.shape[0], bs):
        pk = torch.from_numpy(packed[i:i+bs].astype(np.int64)).to(dev)
        h = tuple(x[i:i+bs].to(dev) for x in hist) if hist is not None else None
        cls, sq = model.encode(pk, hist=h)
        C.append(cls.float().cpu()); S.append(sq.float().mean(1).cpu())
    return torch.cat(C), torch.cat(S)


def train_head(X, Y, dev, out_dim, epochs=30, seed=0):
    torch.manual_seed(seed)
    head = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, out_dim)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    X = X.to(dev); Y = Y.to(dev)
    for ep in range(epochs):
        head.train(); p = torch.randperm(len(X), device=dev)
        for i in range(0, len(X), 4096):
            b = p[i:i+4096]
            logit = head(X[b])
            loss = (torch.nn.functional.cross_entropy(logit, Y[b]) if out_dim > 1
                    else torch.nn.functional.binary_cross_entropy_with_logits(logit.squeeze(1), Y[b].float()))
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval(); return head


def auc_score(scores, labels):  # rank-based AUC (labels 1=pos,0=neg)
    order = np.argsort(scores); ranks = np.empty_like(order, dtype=np.float64); ranks[order] = np.arange(1, len(scores)+1)
    npos = labels.sum(); nneg = len(labels) - npos
    return (ranks[labels == 1].sum() - npos*(npos+1)/2) / (npos*nneg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/eloquence_bulk/databases/wdl_history_128M.h5")
    ap.add_argument("--ckpt", default="style_policy_checkpoints/multiband_history_128M_big/multiband_history_128M_big.pt")
    ap.add_argument("--per-band", type=int, default=12000)
    ap.add_argument("--nat-n", type=int, default=80000)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); dev = a.device; rng = np.random.default_rng(a.seed); random.seed(a.seed)
    model, n_ply = load_encoder(a.ckpt, dev)
    f = h5py.File(a.data, "r")
    elo = f["elo_to_move"][:]
    band = np.clip((elo // 100) * 100, 1000, 2100)

    # ---------- (A) ELO-band classifier ----------
    sel = []
    for b in BANDS:
        ib = np.where(band == b)[0]
        sel.append(rng.choice(ib, min(a.per_band, len(ib)), replace=False))
    idx = np.sort(np.concatenate(sel))
    y = np.array([BANDS.index(int(np.clip((elo[i]//100)*100, 1000, 2100))) for i in idx])
    packed = f["packed_pre"][idx]
    hist = None
    if n_ply:
        hist = (torch.from_numpy(f["hist_from"][idx][:, :n_ply].astype(np.int64)),
                torch.from_numpy(f["hist_to"][idx][:, :n_ply].astype(np.int64)),
                torch.from_numpy(f["hist_cap"][idx][:, :n_ply].astype(np.int64)))
    print(f"[A] encoding {len(idx):,} positions across {len(BANDS)} bands ...", flush=True)
    cls, meansq = encode_feats(model, packed, hist, dev)
    perm = rng.permutation(len(idx)); ntr = int(0.8 * len(idx))
    tri, tei = perm[:ntr], perm[ntr:]
    yte = torch.from_numpy(y[tei])
    feats = {"cls": cls, "meansq": meansq, "cls+sq": torch.cat([cls, meansq], 1)}
    print(f"[A] ELO-BAND classifier (test n={len(tei):,}, chance {100/12:.1f}%):")
    for name, F in feats.items():
        clf = train_head(F[tri], torch.from_numpy(y[tri]), dev, 12)
        with torch.no_grad():
            pred = clf(F[tei].to(dev)).argmax(1).cpu()
        acc = (pred == yte).float().mean().item(); w1 = (pred - yte).abs().le(1).float().mean().item()
        mabe = (pred - yte).abs().float().mean().item()
        print(f"    [{name:7s}] exact={100*acc:.1f}%  within-1={100*w1:.1f}%  mean|band err|={mabe:.2f} (~{mabe*100:.0f} elo)")

    # ---------- (B) naturalness: human vs random afterstate ----------
    nsel = np.sort(idx[rng.choice(len(idx), min(a.nat_n, len(idx)), replace=False)])
    pk = f["packed_pre"][nsel]; fs = f["from_sq"][nsel]; ts = f["to_sq"][nsel]
    hum, rnd = [], []
    for j in range(len(nsel)):
        board = packed_to_board(pk[j].astype(np.uint8))
        legal = list(board.legal_moves)
        if len(legal) < 2:
            continue
        hm = chess.Move(int(fs[j]), int(ts[j]))
        if hm not in legal:
            hm = chess.Move(int(fs[j]), int(ts[j]), promotion=chess.QUEEN)
        if hm not in legal:
            continue
        rm = random.choice([m for m in legal if m != hm])
        bh = board.copy(); bh.push(hm); hum.append(board_to_packed(bh))
        br = board.copy(); br.push(rm); rnd.append(board_to_packed(br))
    hum = np.asarray(hum); rnd = np.asarray(rnd)
    print(f"[B] encoding {len(hum):,} human + {len(rnd):,} random afterstates (hist=None) ...", flush=True)
    ch_c, ch_s = encode_feats(model, hum, None, dev); cr_c, cr_s = encode_feats(model, rnd, None, dev)
    Y = torch.cat([torch.ones(len(ch_c)), torch.zeros(len(cr_c))])
    pn = rng.permutation(len(Y)); nt = int(0.8 * len(Y)); ptri, ptei = pn[:nt], pn[nt:]
    natf = {"cls": (torch.cat([ch_c, cr_c])), "cls+sq": torch.cat([torch.cat([ch_c, ch_s], 1), torch.cat([cr_c, cr_s], 1)])}
    yb = Y[ptei].numpy()
    print(f"[B] NATURALNESS (human=1 vs random=0), test n={len(ptei):,}:")
    for name, X in natf.items():
        disc = train_head(X[ptri], Y[ptri], dev, 1)
        with torch.no_grad():
            s = torch.sigmoid(disc(X[ptei].to(dev)).squeeze(1)).cpu().numpy()
        dacc = ((s > 0.5).astype(float) == yb).mean()
        print(f"    [{name:7s}] acc={100*dacc:.1f}%  AUC={auc_score(s, yb):.3f}  "
              f"P(human): human-af={s[yb==1].mean():.3f} random-af={s[yb==0].mean():.3f}")


if __name__ == "__main__":
    main()
