"""Does SAMPLING the move-predictor stay on the human state-manifold, or drift OOD?
Self-play the policy (bot vs bot at a band), collect reached mid-game states, and ask whether a
discriminator (on the FROZEN encoder's features) can tell them apart from REAL human game states.

Calibrated with controls:
  human-A vs human-B  -> AUC floor (~0.5: two human samples are indistinguishable)
  human vs SAMPLED    -> the question (low = sampling self-stabilizes; high = drift)
  human vs ARGMAX     -> expect higher (greedy drifts more)
  human vs RANDOM     -> AUC ceiling (clearly off-manifold)
Discriminator features are position-only (hist=None) so it judges the STATE, not the trajectory.
Self-play move generation DOES use 2-ply history (realistic play). 128M big model, GPU.
"""
from __future__ import annotations
import argparse, numpy as np, torch, h5py, chess
from collections import deque
from style_policy.multiband_policy import MultiBandPolicy
from style_policy.board_encode import board_to_packed
_NEG = float("-inf")


def load(ckpt, dev):
    ck = torch.load(ckpt, map_location=dev); arch = ck["architecture"]
    m = MultiBandPolicy.from_config(arch).to(dev).eval(); m.load_state_dict(ck["model"])
    n_ply = int(arch.get("n_history_ply", 0)) if arch.get("use_last_move") else 0
    return m, n_ply


def _hist_tensors(recents, n_ply, dev):
    hf, ht, hc = [], [], []
    for rc in recents:
        r = list(rc)[::-1]  # newest-first
        f = [r[i][0] if i < len(r) else -1 for i in range(n_ply)]
        t = [r[i][1] if i < len(r) else -1 for i in range(n_ply)]
        c = [r[i][2] if i < len(r) else 0 for i in range(n_ply)]
        hf.append(f); ht.append(t); hc.append(c)
    return (torch.tensor(hf, device=dev), torch.tensor(ht, device=dev), torch.tensor(hc, device=dev))


def _cap(board, mv):
    if not board.is_capture(mv): return 0
    if board.is_en_passant(mv): return 1
    p = board.piece_at(mv.to_square); return p.piece_type if p else 0


@torch.no_grad()
def selfplay(model, head, n_ply, n_games, max_ply, mode, dev, collect_from=8, stride=4, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    boards = [chess.Board() for _ in range(n_games)]
    recents = [deque(maxlen=max(n_ply, 1)) for _ in range(n_games)]
    out = []
    for ply in range(max_ply):
        act = [i for i in range(n_games) if not boards[i].is_game_over()]
        if not act: break
        packed = np.stack([board_to_packed(boards[i]) for i in act])
        hist = _hist_tensors([recents[i] for i in act], n_ply, dev) if n_ply else None
        cls, sq = model.encode(torch.from_numpy(packed.astype(np.int64)).to(dev), hist=hist)
        for k, i in enumerate(act):
            b = boards[i]
            by = {}
            for m in b.legal_moves:
                key = (m.from_square, m.to_square)
                if key not in by or m.promotion == chess.QUEEN: by[key] = m
            froms = sorted({f for f, _ in by})
            fl = head.from_logits(sq[k:k+1], cls[k:k+1])[0][froms]
            fi = torch.multinomial(torch.softmax(fl, -1), 1, generator=g).item() if mode == "sample" else int(fl.argmax())
            f = froms[fi]
            tos = [t for (ff, t) in by if ff == f]
            tl = head.to_logits(sq[k:k+1], torch.tensor([f], device=dev), cls[k:k+1])[0][tos]
            ti = torch.multinomial(torch.softmax(tl, -1), 1, generator=g).item() if mode == "sample" else int(tl.argmax())
            mv = by[(f, tos[ti])]
            recents[i].append((f, tos[ti], _cap(b, mv)))
            b.push(mv)
            if ply >= collect_from and (ply % stride == 0):
                out.append(board_to_packed(b))
    return np.array(out)


@torch.no_grad()
def feats(model, packed, dev, bs=256):
    C, S = [], []
    for i in range(0, len(packed), bs):
        pk = torch.from_numpy(np.asarray(packed[i:i+bs]).astype(np.int64)).to(dev)
        c, s = model.encode(pk, hist=None); C.append(c.float().cpu()); S.append(s.float().mean(1).cpu())
    return torch.cat([torch.cat(C), torch.cat(S)], 1)


def auc(scores, labels):
    o = np.argsort(scores); r = np.empty_like(o, float); r[o] = np.arange(1, len(scores)+1)
    npos = labels.sum(); return (r[labels == 1].sum() - npos*(npos+1)/2) / (npos*(len(labels)-npos))


def disc_auc(Fpos, Fneg, dev, seed=0):
    torch.manual_seed(seed)
    X = torch.cat([Fpos, Fneg]); Y = torch.cat([torch.ones(len(Fpos)), torch.zeros(len(Fneg))])
    p = torch.randperm(len(X)); ntr = int(0.8*len(X)); tri, tei = p[:ntr], p[ntr:]
    h = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 128), torch.nn.ReLU(), torch.nn.Linear(128, 1)).to(dev)
    opt = torch.optim.AdamW(h.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(25):
        pp = tri[torch.randperm(len(tri))]
        for i in range(0, len(tri), 2048):
            b = pp[i:i+2048]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(h(X[b].to(dev)).squeeze(1), Y[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        s = h(X[tei].to(dev)).squeeze(1).cpu().numpy()
    return auc(s, Y[tei].numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="style_policy_checkpoints/multiband_history_128M_big/multiband_history_128M_big.pt")
    ap.add_argument("--val", default="/mnt/eloquence_bulk/databases/wdl_validation_2025_05.h5")
    ap.add_argument("--band", type=int, default=1500)
    ap.add_argument("--n-games", type=int, default=256); ap.add_argument("--max-ply", type=int, default=60)
    ap.add_argument("--device", default="cuda"); a = ap.parse_args(); dev = a.device
    model, n_ply = load(a.ckpt, dev)
    head = model.heads[int(model.head_index(torch.tensor([a.band])).item())]

    print("self-play (sample) ...", flush=True)
    sp_sample = selfplay(model, head, n_ply, a.n_games, a.max_ply, "sample", dev, seed=1)
    print(f"  {len(sp_sample):,} states", flush=True)
    print("self-play (argmax) ...", flush=True)
    sp_argmax = selfplay(model, head, n_ply, a.n_games, a.max_ply, "argmax", dev, seed=2)
    print(f"  {len(sp_argmax):,} states", flush=True)
    print("random play ...", flush=True)
    # random-play via sampling a fresh (untrained-ish) uniform policy: reuse selfplay but override by random moves
    def randplay(n_games, max_ply, seed):
        rng = np.random.default_rng(seed); boards = [chess.Board() for _ in range(n_games)]; out = []
        for ply in range(max_ply):
            for i, b in enumerate(boards):
                if b.is_game_over(): continue
                mv = rng.choice(list(b.legal_moves)); b.push(mv)
                if ply >= 8 and ply % 4 == 0: out.append(board_to_packed(b))
        return np.array(out)
    rnd = randplay(a.n_games, a.max_ply, 3)

    # human states at the band (packed_pre from val)
    f = h5py.File(a.val, "r"); elo = f["elo_to_move"][:]
    idx = np.where((elo >= a.band) & (elo < a.band + 100))[0]
    rng = np.random.default_rng(0); idx = np.sort(rng.choice(idx, min(len(sp_sample)*2, len(idx)), replace=False))
    human = f["packed_pre"][idx]
    nh = len(human) // 2
    print(f"human states: {len(human):,} (band {a.band})", flush=True)

    print("encoding + discriminators ...", flush=True)
    Fh = feats(model, human, dev); Fss = feats(model, sp_sample, dev)
    Fsa = feats(model, sp_argmax, dev); Frn = feats(model, rnd, dev)
    print(f"\n===== SELF-PLAY DRIFT (band {a.band}, discriminator AUC vs human states) =====")
    print(f"  human-A vs human-B  (floor): {disc_auc(Fh[:nh], Fh[nh:], dev):.3f}")
    print(f"  human vs SAMPLED    (test):  {disc_auc(Fh, Fss, dev):.3f}")
    print(f"  human vs ARGMAX     (greedy):{disc_auc(Fh, Fsa, dev):.3f}")
    print(f"  human vs RANDOM     (ceil):  {disc_auc(Fh, Frn, dev):.3f}")
    print("  read: SAMPLED near floor => sampling self-stabilizes; near ceil => drifts OOD")


if __name__ == "__main__":
    main()
