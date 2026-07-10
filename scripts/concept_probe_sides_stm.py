"""Absolute-vs-STM both-sides probe: does the mover/opponent balance differ between our ABSOLUTE
encoder (128M-big, a1=0 + turn token) and an STM-canonical encoder (Maia-3 backbone, mirror-for-black
+ own/opp channels)? Same side-specific concepts + labels as concept_probe_sides; mean-sq pooling for
BOTH (Maia-3 has no CLS) to isolate the frame. Labels are frame-invariant counts (mirroring doesn't
change how many of the mover's pieces hang), so they apply to both encoders unchanged. CPU by default.

CAVEAT: Maia-3 differs from ours in MANY ways (arch/data/elo+history/size/GAB), so a delta difference
is SUGGESTIVE of a frame effect, not a controlled ablation. The clean test is same-arch both frames."""
from __future__ import annotations
import argparse, sys, os, numpy as np, torch, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concept_probe import load_encoder, encode, train_probe
from concept_probe_sides import hanging, king_safety, mobility, doubled, isolated
from maia3_backbone_probe import build_maia3, maia3_backbone_feats
from style_policy.board_encode import packed_to_board

CONCEPTS = [("hanging_pieces", hanging), ("king_safety", king_safety),
            ("mobility", mobility), ("doubled_pawns", doubled), ("isolated_pawns", isolated)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--our-ckpt", default="style_policy_checkpoints/multiband_history_128M_big/multiband_history_128M_big.pt")
    ap.add_argument("--maia3-model", default="maia3-23m")
    ap.add_argument("--val", default="/mnt/eloquence_bulk/databases/wdl_validation_2025_05.h5")
    ap.add_argument("--n", type=int, default=8000); ap.add_argument("--device", default="cpu")
    a = ap.parse_args(); dev = a.device
    f = h5py.File(a.val, "r"); N = f["packed_pre"].shape[0]
    idx = np.sort(np.random.default_rng(0).choice(N, a.n, replace=False))
    packed = f["packed_pre"][idx]
    print(f"reconstructing {a.n:,} boards ...", flush=True)
    boards = [packed_to_board(p.astype(np.uint8)) for p in packed]

    print("encoding OURS (absolute, mean-sq) ...", flush=True)
    m = load_encoder(a.our_ckpt, dev); _, sq = encode(m, packed, dev); OUR = sq.mean(1)
    print("encoding MAIA-3 backbone (STM, mean-sq) ...", flush=True)
    eng, store = build_maia3(a.maia3_model, dev); M3 = maia3_backbone_feats(eng, store, boards, dev)

    print(f"\n===== ABSOLUTE (ours) vs STM (Maia-3) BOTH-SIDES PROBE (n={a.n:,}, mean-sq) =====")
    hdr = f"{'concept':<15}|{'OUR mov':>8}{'OUR opp':>8}{'OUR d':>8}  |{'M3 mov':>8}{'M3 opp':>8}{'M3 d':>8}"
    print(hdr); print("-"*len(hdr))
    for name, fn in CONCEPTS:
        ymov = np.array([fn(b, b.turn) for b in boards], dtype=object)
        yopp = np.array([fn(b, not b.turn) for b in boards], dtype=object)
        keep = np.array([(mv is not None and op is not None) for mv, op in zip(ymov, yopp)])
        ki = torch.from_numpy(np.where(keep)[0])
        ym = torch.tensor(ymov[keep].astype(np.float32)); yo = torch.tensor(yopp[keep].astype(np.float32))
        om, oo = train_probe(OUR[ki], ym, "reg", dev), train_probe(OUR[ki], yo, "reg", dev)
        mm, mo = train_probe(M3[ki],  ym, "reg", dev), train_probe(M3[ki],  yo, "reg", dev)
        print(f"{name:<15}|{om:>8.3f}{oo:>8.3f}{om-oo:>8.3f}  |{mm:>8.3f}{mo:>8.3f}{mm-mo:>8.3f}", flush=True)
    print("\n  d = mover-opp R2. Compare the two 'd' columns: if STM's deltas are systematically more")
    print("  positive (mover-favored) than absolute's, STM privileges the side-to-move; if similar,")
    print("  the mover/opp balance is task-driven, not frame-driven. (absolute values not comparable")
    print("  across encoders — diff arch/data/size; only the WITHIN-encoder deltas are.)")


if __name__ == "__main__":
    main()
