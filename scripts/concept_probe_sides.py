"""Both-sides concept probe: does the encoder decode the SIDE-TO-MOVE's features as well as the
OPPONENT's? Move-prediction trains on the mover's move, so the encoder may over-represent the
mover's situation. mover >> opponent => mover-biased (turn-agnostic encoder would enrich the world
model); ~equal => already both-sides complete. Global features [CLS ++ mean-sq]. CPU by default
(so it doesn't contend with a training GPU)."""
from __future__ import annotations
import argparse, sys, os, numpy as np, torch, h5py, chess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concept_probe import load_encoder, encode, train_probe
from style_policy.board_encode import packed_to_board


def hanging(b, side):
    opp = not side
    return float(sum(1 for sq, p in b.piece_map().items()
                     if p.color == side and b.attackers(opp, sq) and not b.attackers(side, sq)))

def king_safety(b, side):
    ks = b.king(side)
    if ks is None: return 0.0
    sqs = [ks] + list(chess.SquareSet(chess.BB_KING_ATTACKS[ks]))
    return float(sum(len(b.attackers(not side, s)) for s in sqs))

def mobility(b, side):
    if b.turn == side: return float(b.legal_moves.count())
    if b.is_check(): return None  # opponent mobility undefined while side-to-move is in check
    b2 = b.copy(stack=False); b2.push(chess.Move.null())
    return float(b2.legal_moves.count())

def _pfiles(b, side):
    f = [0]*8
    for sq, p in b.piece_map().items():
        if p.color == side and p.piece_type == chess.PAWN: f[chess.square_file(sq)] += 1
    return f
def doubled(b, side):  return float(sum(max(0, x-1) for x in _pfiles(b, side)))
def isolated(b, side):
    f = _pfiles(b, side); c = 0
    for i in range(8):
        if f[i] and (i == 0 or f[i-1] == 0) and (i == 7 or f[i+1] == 0): c += f[i]
    return float(c)

CONCEPTS = [("hanging_pieces", hanging), ("king_safety", king_safety),
            ("mobility", mobility), ("doubled_pawns", doubled), ("isolated_pawns", isolated)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="style_policy_checkpoints/multiband_history_128M_big/multiband_history_128M_big.pt")
    ap.add_argument("--val", default="/mnt/eloquence_bulk/databases/wdl_validation_2025_05.h5")
    ap.add_argument("--n", type=int, default=12000); ap.add_argument("--device", default="cpu")
    a = ap.parse_args(); dev = a.device
    f = h5py.File(a.val, "r"); N = f["packed_pre"].shape[0]
    idx = np.sort(np.random.default_rng(0).choice(N, a.n, replace=False))
    packed = f["packed_pre"][idx]
    print(f"reconstructing {a.n:,} boards ...", flush=True)
    boards = [packed_to_board(p.astype(np.uint8)) for p in boards_src(packed)]
    print("encoding (CPU) ...", flush=True)
    m = load_encoder(a.ckpt, dev); cls, sq = encode(m, packed, dev)
    G = torch.cat([cls, sq.mean(1)], 1)  # [CLS ++ mean-sq]

    print(f"\n===== BOTH-SIDES CONCEPT PROBE: {a.ckpt.split('/')[-1]} (n={a.n:,}) =====")
    print(f"{'concept':<15}{'MOVER R2':>10}{'OPPONENT R2':>12}{'delta(mover-opp)':>18}")
    for name, fn in CONCEPTS:
        ymov = np.array([fn(b, b.turn) for b in boards], dtype=object)
        yopp = np.array([fn(b, not b.turn) for b in boards], dtype=object)
        keep = np.array([ (mv is not None and op is not None) for mv, op in zip(ymov, yopp) ])
        Gk = G[torch.from_numpy(np.where(keep)[0])]
        ym = torch.tensor(ymov[keep].astype(np.float32)); yo = torch.tensor(yopp[keep].astype(np.float32))
        rm = train_probe(Gk, ym, "reg", dev); ro = train_probe(Gk, yo, "reg", dev)
        print(f"{name:<15}{rm:>10.3f}{ro:>12.3f}{rm-ro:>18.3f}", flush=True)
    print("\n  R2 on [CLS++mean-sq]. mover>>opp => encoder is mover-biased (turn-agnostic would enrich);")
    print("  ~equal => encoder already represents both sides (turn-aware is fine).")


def boards_src(packed):
    return packed  # (kept as a hook; packed rows iterated directly)


if __name__ == "__main__":
    main()
