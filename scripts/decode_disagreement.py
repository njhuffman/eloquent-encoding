#!/usr/bin/env python3
"""How often does factored-argmax (argmax P(from) then argmax P(to|from)) disagree with the true
joint-argmax (argmax over legal moves of P(from)*P(to|from))? And does joint decoding change the
human-move top-1 match? Quantifies whether the factored-decoding bug matters. Uses the multiband
model with history K=2."""
from __future__ import annotations
import argparse
import numpy as np
import torch
import h5py
import chess
from style_policy.multiband_policy import MultiBandPolicy
from style_policy.board_encode import packed_to_board


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="style_policy_checkpoints/multiband_history_128M_big/multiband_history_128M_big.pt")
    ap.add_argument("--val-h5", default="/mnt/eloquence_bulk/databases/wdl_validation_2025_05.h5")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    ck = torch.load(a.ckpt, map_location=dev)
    model = MultiBandPolicy.from_config(ck["architecture"]); model.load_state_dict(ck["model"]); model.to(dev).eval()

    with h5py.File(a.val_h5, "r") as f:
        n = min(a.n, int(f["packed_pre"].shape[0]))
        packed = f["packed_pre"][:n]
        from_sq = f["from_sq"][:n].astype(int); to_sq = f["to_sq"][:n].astype(int); elo = f["elo_to_move"][:n].astype(int)
        hf = f["hist_from"][:n].astype(np.int64); ht = f["hist_to"][:n].astype(np.int64); hc = f["hist_cap"][:n].astype(np.int64)

    disagree = 0; considered = 0
    fac_hit = 0; joint_hit = 0
    for i in range(n):
        board = packed_to_board(np.asarray(packed[i], np.uint8))
        if board.is_game_over():
            continue
        legal = list(board.legal_moves)
        froms = sorted({m.from_square for m in legal})
        if len(legal) < 2:
            continue
        considered += 1
        pk = torch.from_numpy(np.asarray(packed[i], np.uint8)[None]).to(dev)
        hist = (torch.tensor([hf[i]], device=dev), torch.tensor([ht[i]], device=dev), torch.tensor([hc[i]], device=dev))
        cls, sq = model.encode(pk, hist=hist)
        g = int(model.head_index(torch.tensor([elo[i]])).item()); head = model.heads[g]
        fl = head.from_logits(sq, cls)[0]                                   # (64,)
        pfrom = torch.softmax(fl[froms], dim=-1)                            # over legal froms
        # to|from for every legal from (batched)
        sqe = sq.expand(len(froms), -1, -1); cle = cls.expand(len(froms), -1)
        tl = head.to_logits(sqe, torch.tensor(froms, device=dev), cle)      # (F,64)
        best = None  # (jointprob, from, to)
        fac_from_i = int(pfrom.argmax()); fac_from = froms[fac_from_i]
        fac_to = None; fac_to_p = -1.0
        for fi, fsq in enumerate(froms):
            tos = [m.to_square for m in legal if m.from_square == fsq]
            pto = torch.softmax(tl[fi][tos], dim=-1)
            pf = float(pfrom[fi])
            # factored to (only for the factored-chosen from)
            if fsq == fac_from:
                ti = int(pto.argmax()); fac_to = tos[ti]
            # joint over this from's moves
            j_ti = int((pf * pto).argmax()); j_p = pf * float(pto[j_ti])
            if best is None or j_p > best[0]:
                best = (j_p, fsq, tos[j_ti])
        jf, jt = best[1], best[2]
        if (jf, jt) != (fac_from, fac_to):
            disagree += 1
        if fac_from == from_sq[i] and fac_to == to_sq[i]:
            fac_hit += 1
        if jf == from_sq[i] and jt == to_sq[i]:
            joint_hit += 1

    print(f"n_considered={considered}")
    print(f"factored vs joint DISAGREE: {disagree} ({100*disagree/considered:.2f}%)")
    print(f"human-move top-1:  factored={100*fac_hit/considered:.2f}%   joint={100*joint_hit/considered:.2f}%   "
          f"(joint - factored = {100*(joint_hit-fac_hit)/considered:+.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
