"""Concept-probing suite: how well does a frozen encoder LINEARLY encode chess concepts?
The reusable "encoder quality" benchmark. Ground-truth labels computed from the board (python-chess).

Robustness controls (a probe score alone is meaningless):
  - TRAINED   : linear probe on the frozen trained encoder's features.
  - RANDOM    : same probe on a same-architecture RANDOM-INIT encoder (untrained-features floor).
  - RAW-INPUT : linear probe on the raw board planes (is the concept just linearly in the input?).
The encoder earns credit on a concept only where TRAINED >> RANDOM and TRAINED >> RAW.

Concept ladder: input-preserved (material, occupancy, side-to-move — sanity floors) ->
derived (mobility, hanging pieces, king-attackers, doubled/isolated pawns, per-square attack maps).
Global concepts probe [CLS ++ mean-squares]; per-square concepts probe the 64 square tokens.
"""
from __future__ import annotations
import argparse, numpy as np, torch, h5py, chess
from style_policy.multiband_policy import MultiBandPolicy
from style_policy.board_encode import packed_to_board

PVAL = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}


# ---------- concept label functions (board -> scalar | 0/1 | (64,) array) ----------
def _own_pawns_files(b, color):
    f = [0]*8
    for sq, p in b.piece_map().items():
        if p.color == color and p.piece_type == chess.PAWN: f[chess.square_file(sq)] += 1
    return f

def lbl_material(b):   # stm perspective
    return float(sum(PVAL[p.piece_type]*(1 if p.color == b.turn else -1) for p in b.piece_map().values()))
def lbl_stm(b):        return 1 if b.turn == chess.WHITE else 0
def lbl_incheck(b):    return 1 if b.is_check() else 0
def lbl_mobility(b):   return float(b.legal_moves.count())
def lbl_hanging(b):
    stm, opp = b.turn, not b.turn; c = 0
    for sq, p in b.piece_map().items():
        if p.color == stm and b.attackers(opp, sq) and not b.attackers(stm, sq): c += 1
    return float(c)
def lbl_kingsafety(b):
    ks = b.king(b.turn)
    if ks is None: return 0.0
    sqs = [ks] + list(chess.SquareSet(chess.BB_KING_ATTACKS[ks]))
    return float(sum(len(b.attackers(not b.turn, s)) for s in sqs))
def lbl_doubled(b):
    return float(sum(max(0, f-1) for f in _own_pawns_files(b, b.turn)))
def lbl_isolated(b):
    f = _own_pawns_files(b, b.turn); c = 0
    for i in range(8):
        if f[i] and (i == 0 or f[i-1] == 0) and (i == 7 or f[i+1] == 0): c += f[i]
    return float(c)
def lbl_attacked_white(b):  # per-square: is square attacked by white
    return np.array([1 if b.attackers(chess.WHITE, s) else 0 for s in range(64)], np.int64)
def lbl_occupied(b):        # per-square floor
    return np.array([1 if b.piece_at(s) else 0 for s in range(64)], np.int64)

CONCEPTS = [
    ("material",       "global", "reg", lbl_material,      "input-preserved floor"),
    ("side_to_move",   "global", "bin", lbl_stm,           "trivial floor"),
    ("in_check",       "global", "bin", lbl_incheck,       "derived (king attacked)"),
    ("mobility",       "global", "reg", lbl_mobility,      "derived (legal moves)"),
    ("hanging_pieces", "global", "reg", lbl_hanging,       "derived (tactical)"),
    ("king_safety",    "global", "reg", lbl_kingsafety,    "derived (positional)"),
    ("doubled_pawns",  "global", "reg", lbl_doubled,       "derived (structure)"),
    ("isolated_pawns", "global", "reg", lbl_isolated,      "derived (structure)"),
    ("attacked_by_W",  "square", "bin", lbl_attacked_white,"derived per-square (attack map)"),
    ("occupied",       "square", "bin", lbl_occupied,      "input-preserved per-square floor"),
]


def board_planes(b):  # (768,) raw-input: 12 piece-type-color planes x 64
    pl = np.zeros((12, 64), np.float32)
    for sq, p in b.piece_map().items():
        pl[(p.piece_type-1) + (0 if p.color == chess.WHITE else 6), sq] = 1.0
    return pl.reshape(-1)


def load_encoder(ckpt, dev, random_init=False):
    ck = torch.load(ckpt, map_location=dev); arch = ck["architecture"]
    m = MultiBandPolicy.from_config(arch).to(dev).eval()
    if not random_init: m.load_state_dict(ck["model"])
    return m


@torch.no_grad()
def encode(model, packed, dev, bs=256):
    C, S = [], []
    for i in range(0, len(packed), bs):
        pk = torch.from_numpy(packed[i:i+bs].astype(np.int64)).to(dev)
        c, s = model.encode(pk, hist=None); C.append(c.float().cpu()); S.append(s.float().cpu())
    return torch.cat(C), torch.cat(S)


def train_probe(X, y, kind, dev, epochs=60):
    """Linear probe; returns held-out metric (R2 for reg, accuracy for bin)."""
    n = len(X); rng = torch.randperm(n); tr, te = rng[:int(0.8*n)], rng[int(0.8*n):]
    out_dim = 1 if kind == "reg" else 2
    W = torch.nn.Linear(X.shape[1], out_dim).to(dev)
    opt = torch.optim.AdamW(W.parameters(), lr=1e-2, weight_decay=1e-4)
    Xtr, Xte = X[tr].to(dev), X[te].to(dev)
    if kind == "reg":
        ytr = y[tr].float().to(dev); yte = y[te].float().to(dev)
        mu, sd = ytr.mean(), ytr.std().clamp_min(1e-6)
        for _ in range(epochs):
            for i in range(0, len(Xtr), 8192):
                loss = ((W(Xtr[i:i+8192]).squeeze(1) - (ytr[i:i+8192]-mu)/sd)**2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pred = W(Xte).squeeze(1)*sd + mu
            ss_res = ((pred-yte)**2).sum(); ss_tot = ((yte-yte.mean())**2).sum().clamp_min(1e-6)
            return float(1 - ss_res/ss_tot)
    else:
        ytr = y[tr].to(dev); yte = y[te].to(dev)
        for _ in range(epochs):
            for i in range(0, len(Xtr), 8192):
                loss = torch.nn.functional.cross_entropy(W(Xtr[i:i+8192]), ytr[i:i+8192])
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            return float((W(Xte).argmax(1) == yte).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="style_policy_checkpoints/multiband_history_128M_big/multiband_history_128M_big.pt")
    ap.add_argument("--val", default="/mnt/eloquence_bulk/databases/wdl_validation_2025_05.h5")
    ap.add_argument("--n", type=int, default=20000); ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); dev = a.device
    f = h5py.File(a.val, "r"); N = f["packed_pre"].shape[0]
    idx = np.sort(np.random.default_rng(0).choice(N, a.n, replace=False))
    packed = f["packed_pre"][idx]
    print(f"reconstructing boards + labels for {a.n:,} positions ...", flush=True)
    boards = [packed_to_board(p.astype(np.uint8)) for p in packed]
    labels = {name: (np.array([fn(b) for b in boards]) if kind == "global"
                     else np.stack([fn(b) for b in boards]))
              for (name, kind, _, fn, _) in CONCEPTS}
    raw_planes = torch.from_numpy(np.stack([board_planes(b) for b in boards]))  # (N,768)

    print("encoding (trained + random-init) ...", flush=True)
    mt = load_encoder(a.ckpt, dev); cT, sT = encode(mt, packed, dev); del mt; torch.cuda.empty_cache()
    mr = load_encoder(a.ckpt, dev, random_init=True); cR, sR = encode(mr, packed, dev); del mr; torch.cuda.empty_cache()
    globT = torch.cat([cT, sT.mean(1)], 1); globR = torch.cat([cR, sR.mean(1)], 1)  # [CLS ++ mean-sq]

    print(f"\n===== CONCEPT PROBE: {a.ckpt.split('/')[-1]} (n={a.n:,}) =====")
    print(f"{'concept':<15}{'kind':<10}{'trained':>9}{'random':>9}{'raw-inp':>9}   note")
    for (name, kind, tgt, _, note) in CONCEPTS:
        y = torch.from_numpy(labels[name])
        if kind == "global":
            sc_t = train_probe(globT, y, tgt, dev)
            sc_r = train_probe(globR, y, tgt, dev)
            sc_raw = train_probe(raw_planes, y, tgt, dev)
        else:  # per-square: flatten (N,64,d)->(N*64,d), labels (N,64)->(N*64,)
            yf = y.reshape(-1)
            sc_t = train_probe(sT.reshape(-1, sT.shape[-1]), yf, tgt, dev)
            sc_r = train_probe(sR.reshape(-1, sR.shape[-1]), yf, tgt, dev)
            # raw per-square baseline: the 12-dim piece one-hot at each square (can't see other squares)
            rawsq = raw_planes.reshape(len(boards), 12, 64).permute(0, 2, 1).reshape(-1, 12)
            sc_raw = train_probe(rawsq, yf, tgt, dev)
        print(f"{name:<15}{tgt+'/'+kind[:3]:<10}{sc_t:>9.3f}{sc_r:>9.3f}{sc_raw:>9.3f}   {note}", flush=True)
    print("\n  reg=R2, bin=accuracy. Encoder 'understands' a concept where trained >> random AND trained >> raw-inp.")


if __name__ == "__main__":
    main()
