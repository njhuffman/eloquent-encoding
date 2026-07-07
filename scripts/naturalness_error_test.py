"""Does the naturalness head help where the POLICY is wrong?
Train the naturalness disc (human vs random afterstate, cls+sq) on train data, then on held-out
2025-05 positions: get the policy's top-k moves + the human move, score each candidate's afterstate
naturalness, and ask:
  (1) On policy ERRORS (top-1 != human, human in top-k): does naturalness rank the human move's
      afterstate above the policy's wrong pick's afterstate? (>50% => complementary signal)
  (2) Rerank score = policy_logprob + lambda * naturalness_logit(afterstate); sweep lambda; does
      top-1 move-match improve?
"""
from __future__ import annotations
import argparse, numpy as np, torch, h5py, chess, random
from style_policy.multiband_policy import MultiBandPolicy
from style_policy.board_encode import board_to_packed, packed_to_board
BANDS = list(range(1000, 2200, 100)); _NEG = float("-inf")


def load_model(ckpt, dev):
    ck = torch.load(ckpt, map_location=dev); arch = ck["architecture"]
    m = MultiBandPolicy.from_config(arch).to(dev).eval(); m.load_state_dict(ck["model"])
    n_ply = int(arch.get("n_history_ply", 0)) if arch.get("use_last_move") else 0
    return m, n_ply


@torch.no_grad()
def feats(model, packed, dev, hist=None, bs=256):
    C, S = [], []
    for i in range(0, len(packed), bs):
        pk = torch.from_numpy(np.asarray(packed[i:i+bs]).astype(np.int64)).to(dev)
        h = tuple(x[i:i+bs].to(dev) for x in hist) if hist is not None else None
        c, s = model.encode(pk, hist=h); C.append(c.float().cpu()); S.append(s.float().mean(1).cpu())
    return torch.cat([torch.cat(C), torch.cat(S)], 1)  # (N,768) cls+meansq on CPU


def band_idx(elo): return min(11, max(0, int(elo) // 100 - 10))


@torch.no_grad()
def policy_topk(model, head, board, dev, hist, k=5):
    pk = torch.from_numpy(board_to_packed(board)[None].astype(np.int64)).to(dev)
    cls, sq = model.encode(pk, hist=hist)
    by = {}
    for m in board.legal_moves:
        key = (m.from_square, m.to_square)
        if key not in by or m.promotion == chess.QUEEN: by[key] = m
    froms = sorted({fr for fr, _ in by}); fi = {fr: i for i, fr in enumerate(froms)}
    lpf = torch.log_softmax(head.from_logits(sq, cls)[0][froms], 0)
    tl = head.to_logits(sq.expand(len(froms), -1, -1), torch.tensor(froms, device=dev), cls.expand(len(froms), -1))
    tbf = {}
    for fr, t in by: tbf.setdefault(fr, []).append(t)
    lpt = {fr: torch.log_softmax(tl[fi[fr]][tbf[fr]], 0) for fr in froms}
    scored = [(float(lpf[fi[m.from_square]]) + float(lpt[m.from_square][tbf[m.from_square].index(m.to_square)]), m)
              for m in by.values()]
    scored.sort(key=lambda x: -x[0]); return scored[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="style_policy_checkpoints/multiband_history_128M_big/multiband_history_128M_big.pt")
    ap.add_argument("--train-data", default="/mnt/eloquence_bulk/databases/wdl_history_128M.h5")
    ap.add_argument("--test-data", default="/mnt/eloquence_bulk/databases/wdl_validation_2025_05.h5")
    ap.add_argument("--disc-n", type=int, default=60000); ap.add_argument("--test-n", type=int, default=2000)
    ap.add_argument("--k", type=int, default=5); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); dev = a.device; rng = np.random.default_rng(a.seed); random.seed(a.seed)
    model, n_ply = load_model(a.ckpt, dev)

    # ---- train naturalness disc (human vs random afterstate) ----
    ftr = h5py.File(a.train_data, "r")
    sel = np.sort(rng.choice(ftr["packed_pre"].shape[0], a.disc_n, replace=False))
    pk = ftr["packed_pre"][sel]; fs = ftr["from_sq"][sel]; ts = ftr["to_sq"][sel]
    hum, rnd = [], []
    for j in range(len(sel)):
        b = packed_to_board(pk[j].astype(np.uint8)); legal = list(b.legal_moves)
        if len(legal) < 2: continue
        hm = chess.Move(int(fs[j]), int(ts[j]))
        if hm not in legal: hm = chess.Move(int(fs[j]), int(ts[j]), promotion=chess.QUEEN)
        if hm not in legal: continue
        rm = random.choice([m for m in legal if m != hm])
        bh = b.copy(); bh.push(hm); hum.append(board_to_packed(bh))
        br = b.copy(); br.push(rm); rnd.append(board_to_packed(br))
    print(f"disc train: {len(hum):,} human + {len(rnd):,} random afterstates", flush=True)
    Xh = feats(model, hum, dev); Xr = feats(model, rnd, dev)
    X = torch.cat([Xh, Xr]); Y = torch.cat([torch.ones(len(Xh)), torch.zeros(len(Xr))])  # CPU
    disc = torch.nn.Sequential(torch.nn.Linear(768, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1)).to(dev)
    opt = torch.optim.AdamW(disc.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(30):
        p = torch.randperm(len(X))
        for i in range(0, len(X), 4096):
            bi = p[i:i+4096]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(disc(X[bi].to(dev)).squeeze(1), Y[bi].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
    disc.eval(); del X, Xh, Xr

    @torch.no_grad()
    def nat_logit(board_after):
        return float(disc(feats(model, [board_to_packed(board_after)], dev).to(dev)).squeeze())

    # ---- test set: policy top-k + human + afterstate naturalness ----
    fte = h5py.File(a.test_data, "r")
    ti = np.sort(rng.choice(fte["packed_pre"].shape[0], a.test_n, replace=False))
    tp = fte["packed_pre"][ti]; tf = fte["from_sq"][ti]; tt = fte["to_sq"][ti]; te = fte["elo_to_move"][ti]
    thf = fte["hist_from"][ti][:, :n_ply]; tht = fte["hist_to"][ti][:, :n_ply]; thc = fte["hist_cap"][ti][:, :n_ply]
    lambdas = [0.0, 0.5, 1.0, 2.0, 4.0]
    rr_hit = {l: 0 for l in lambdas}; n = 0
    err_total = 0; err_human_in_topk = 0; nat_prefers_human = 0
    gap_err = []; nat_h_corr = []; nat_h_err = []
    for j in range(len(ti)):
        board = packed_to_board(tp[j].astype(np.uint8)); legal = list(board.legal_moves)
        if len(legal) < 2: continue
        hm = chess.Move(int(tf[j]), int(tt[j]))
        if hm not in legal: hm = chess.Move(int(tf[j]), int(tt[j]), promotion=chess.QUEEN)
        if hm not in legal: continue
        hist = None
        if n_ply:
            hist = (torch.tensor([thf[j]], device=dev).long(), torch.tensor([tht[j]], device=dev).long(),
                    torch.tensor([thc[j]], device=dev).long())
        head = model.heads[band_idx(te[j])]
        topk = policy_topk(model, head, board, dev, hist, a.k)
        n += 1
        hkey = (hm.from_square, hm.to_square)
        # naturalness of each candidate afterstate + human afterstate
        nat = {}
        for lp, m in topk:
            bb = board.copy(); bb.push(m); nat[(m.from_square, m.to_square)] = nat_logit(bb)
        bb = board.copy(); bb.push(hm); nh = nat_logit(bb)
        # rerank top-k by policy + lambda*nat
        for l in lambdas:
            best = max(topk, key=lambda x: x[0] + l * nat[(x[1].from_square, x[1].to_square)])[1]
            if (best.from_square, best.to_square) == hkey: rr_hit[l] += 1
        # error analysis
        p_top = topk[0][1]; correct = (p_top.from_square, p_top.to_square) == hkey
        (nat_h_corr if correct else nat_h_err).append(nh)
        if not correct:
            err_total += 1
            if hkey in [(m.from_square, m.to_square) for _, m in topk]:
                err_human_in_topk += 1
            np_top = nat[(p_top.from_square, p_top.to_square)]
            if nh > np_top: nat_prefers_human += 1
            gap_err.append(nh - np_top)
    print(f"\ntested {n} positions | policy top-1 move-match (lambda=0): {100*rr_hit[0.0]/n:.2f}%")
    print("RERANK top-1 move-match by policy + lambda*naturalness:")
    for l in lambdas:
        print(f"  lambda={l}: {100*rr_hit[l]/n:.2f}%  ({(rr_hit[l]-rr_hit[0.0])*100/n:+.2f})")
    print(f"\nERROR analysis (policy top-1 wrong): errors={err_total}  human-in-top{a.k}={err_human_in_topk}")
    print(f"  on errors, nat(human-af) > nat(policy-top-af): {100*nat_prefers_human/max(err_total,1):.1f}%  (>50% => complementary)")
    print(f"  mean nat-logit gap (human - policy-top) on errors: {np.mean(gap_err):+.3f}")
    print(f"  mean nat-logit of human afterstate: correct-cases={np.mean(nat_h_corr):+.3f}  error-cases={np.mean(nat_h_err):+.3f}")


if __name__ == "__main__":
    main()
