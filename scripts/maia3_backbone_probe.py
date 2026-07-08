"""Q3: probe Maia-3's BACKBONE (features before its policy head) with our concept suite, vs OUR
encoder. Answers: does Maia-3 have better GENERAL features, or just a better-tuned move head?

Maia-3 backbone = output of model.transformer -> (B,64,dim_vit) per-square features (history is in
the channels, not the sequence). Mean-pool over the 64 squares -> global feature. Maia-3 runs in a
side-to-move-canonical frame (mirrors for black), so we use ONLY stm-relative global concepts
(frame-invariant: material/mobility/hanging/king-safety/pawn-structure) and compare our encoder with
mean-squares-only pooling (Maia-3 has no CLS). Controls: random-init-ours + raw-board.
"""
from __future__ import annotations
import argparse, os, sys, numpy as np, torch, h5py, chess
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concept_probe import (lbl_material, lbl_incheck, lbl_mobility, lbl_hanging, lbl_kingsafety,
                           lbl_doubled, lbl_isolated, board_planes, load_encoder, encode, train_probe)
from style_policy.board_encode import packed_to_board
from maia3.uci import parse_args as m3_args, Maia3UCIEngine
from maia3.dataset import tokenize_board

# stm-relative, frame-invariant global concepts (the discriminative ones)
GCON = [
    ("material",       "reg", lbl_material),
    ("in_check",       "bin", lbl_incheck),
    ("mobility",       "reg", lbl_mobility),
    ("hanging_pieces", "reg", lbl_hanging),
    ("king_safety",    "reg", lbl_kingsafety),
    ("doubled_pawns",  "reg", lbl_doubled),
    ("isolated_pawns", "reg", lbl_isolated),
]


def build_maia3(model_name, dev):
    cfg = m3_args(["--model", model_name, "--device", dev])
    eng = Maia3UCIEngine(cfg); eng.ensure_model_loaded()
    store = {}
    eng.model.transformer.register_forward_hook(lambda m, i, o: store.__setitem__("x", o.detach()))
    return eng, store


@torch.no_grad()
def maia3_backbone_feats(eng, store, boards, dev, elo=1500, bs=256):
    H = eng.cfg.history
    out = []
    for i in range(0, len(boards), bs):
        toks = []
        for b in boards[i:i+bs]:
            h = deque([tokenize_board(b)]*H, maxlen=H)   # position-only (no real history)
            toks.append(eng._tokens_from_history(h))
        tokens = torch.stack(toks).to(dev)
        se = torch.full((len(toks),), elo, dtype=torch.long, device=dev)
        eng.model(tokens, se, se)                        # hook fills store["x"] = (B,64,dim_vit)
        out.append(store["x"].float().mean(1).cpu())     # mean-pool 64 squares -> global
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--our-ckpt", default="style_policy_checkpoints/multiband_history_128M_big/multiband_history_128M_big.pt")
    ap.add_argument("--maia3-model", default="maia3-23m")
    ap.add_argument("--val", default="/mnt/eloquence_bulk/databases/wdl_validation_2025_05.h5")
    ap.add_argument("--n", type=int, default=20000); ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); dev = a.device
    f = h5py.File(a.val, "r"); N = f["packed_pre"].shape[0]
    idx = np.sort(np.random.default_rng(0).choice(N, a.n, replace=False))
    packed = f["packed_pre"][idx]
    print(f"reconstructing {a.n:,} boards + labels ...", flush=True)
    boards = [packed_to_board(p.astype(np.uint8)) for p in packed]
    labels = {name: np.array([fn(b) for b in boards]) for (name, _, fn) in GCON}
    raw = torch.from_numpy(np.stack([board_planes(b) for b in boards]))

    print("encoding: OUR (mean-sq) + random-init + Maia-3 backbone ...", flush=True)
    mt = load_encoder(a.our_ckpt, dev); _, sqT = encode(mt, packed, dev); ourT = sqT.mean(1); del mt; torch.cuda.empty_cache()
    mr = load_encoder(a.our_ckpt, dev, random_init=True); _, sqR = encode(mr, packed, dev); ourR = sqR.mean(1); del mr; torch.cuda.empty_cache()
    eng, store = build_maia3(a.maia3_model, dev); m3 = maia3_backbone_feats(eng, store, boards, dev)
    print(f"  feature dims: ours(mean-sq)={ourT.shape[1]}  maia3-backbone={m3.shape[1]}", flush=True)

    print(f"\n===== MAIA-3 BACKBONE vs OUR ENCODER (concept probe, n={a.n:,}, mean-sq pooling) =====")
    print(f"{'concept':<15}{'OURS-128M':>10}{'MAIA3-bb':>10}{'random':>9}{'raw':>9}   verdict")
    for (name, tgt, _) in GCON:
        y = torch.from_numpy(labels[name])
        so = train_probe(ourT, y, tgt, dev); sm = train_probe(m3, y, tgt, dev)
        sr = train_probe(ourR, y, tgt, dev); sraw = train_probe(raw, y, tgt, dev)
        verdict = "maia3>ours" if sm - so > 0.02 else ("ours>maia3" if so - sm > 0.02 else "~tie")
        print(f"{name:<15}{so:>10.3f}{sm:>10.3f}{sr:>9.3f}{sraw:>9.3f}   {verdict}", flush=True)
    print("\n  reg=R2, bin=acc. maia3>>ours => Maia-3 has better GENERAL features (not just move head).")
    print("  ~tie/ours>=maia3 => Maia-3's move-pred edge is head/tuning, not the encoder representation.")


if __name__ == "__main__":
    main()
