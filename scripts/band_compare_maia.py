#!/usr/bin/env python3
"""Per-band top-1 move-match for Maia2 on the 2025-05 val, using the SAME factored metric as
history_ksweep (from = argmax over legal from-marginals; to = argmax over legal moves from the
TRUE from-square; move = both correct). Lets Maia line up with our models' per-band numbers.
Maia elo is clamped to [1000,1900] (its supported range) — the 2000-2199 columns use 1900.
"""
from __future__ import annotations
import argparse
import numpy as np
import h5py
import chess
from collections import defaultdict
from style_policy.board_encode import packed_to_board
from style_policy.maia2_bot import load_maia2
from maia2 import inference


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-h5", default="/mnt/eloquence_bulk/databases/wdl_validation_2025_05.h5")
    ap.add_argument("--n", type=int, default=10000)
    a = ap.parse_args()

    maia, prep = load_maia2("rapid", device="cpu")
    with h5py.File(a.val_h5, "r") as f:
        n = min(a.n, int(f["packed_pre"].shape[0]))
        packed = f["packed_pre"][:n]
        from_sq = f["from_sq"][:n].astype(int)
        to_sq = f["to_sq"][:n].astype(int)
        elo = f["elo_to_move"][:n].astype(int)
        opp = f["opp_elo"][:n].astype(int)

    hit = defaultdict(int); cnt = defaultdict(int); H = 0; C = 0
    import time; t0 = time.time()
    for i in range(n):
        if i and i % 1000 == 0:
            print(f"  ...{i}/{n} ({i/(time.time()-t0):.0f}/s)", flush=True)
        board = packed_to_board(np.asarray(packed[i], np.uint8))
        if board.is_game_over():
            continue
        band = int(min(2100, max(1000, (int(elo[i]) // 100) * 100)))
        se = int(min(1900, max(1000, int(elo[i])))); oe = int(min(1900, max(1000, int(opp[i]))))
        mp, _ = inference.inference_each(maia, prep, board.fen(), se, oe)
        legal = list(board.legal_moves)
        # from-marginal argmax over legal
        fromp = defaultdict(float)
        for m in legal:
            fromp[m.from_square] += mp.get(m.uci(), 0.0)
        if not fromp:
            continue
        pf = max(fromp, key=fromp.get)
        # best to given the TRUE from-square (teacher-forced), over legal moves
        cand = [(m.to_square, mp.get(m.uci(), 0.0)) for m in legal if m.from_square == int(from_sq[i])]
        pt = max(cand, key=lambda kv: kv[1])[0] if cand else -1
        ok = (pf == int(from_sq[i])) and (pt == int(to_sq[i]))
        cnt[band] += 1; C += 1
        if ok:
            hit[band] += 1; H += 1

    print(f"\nMaia2 per-band top-1 move-match (matched metric)  n_eval={C}")
    for b in range(1000, 2200, 100):
        if cnt[b]:
            print(f"  {b}-{b+99}: {100*hit[b]/cnt[b]:.1f}  (n={cnt[b]})")
    print(f"  OVERALL: {100*H/max(C,1):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
