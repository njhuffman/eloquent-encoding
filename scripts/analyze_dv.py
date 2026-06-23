#!/usr/bin/env python3
"""ΔV analysis (B1 distribution + B2 disagreement) for the WDL value head.

Per validation position: V(s) (mover's expected score), then ΔV for the human's move and the
model's greedy top move. Value flips sides after a move, so the resulting position's value is
taken from the new side-to-move's perspective (1 - their expected score) and conditioned on the
new side-to-move's elo (opp_elo).
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import chess
from style_policy.model import BasePolicy
from style_policy.dataset import PackedMoveDataset
from style_policy.model_spec import elo_to_bucket
from style_policy.board_encode import (board_to_packed, packed_to_board,
                                       legal_from_u64, legal_to_u64)
from style_policy.legal_mask import u64_to_mask

_NEG = float("-inf")


def escore(wdl_logits: torch.Tensor) -> float:
    p = torch.softmax(wdl_logits, dim=-1)[0]  # [loss, draw, win]
    return float(p[2] + 0.5 * p[1])


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-h5", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=4000)
    args = ap.parse_args()
    dev = args.device
    ck = torch.load(args.checkpoint, map_location=dev)
    m = BasePolicy.from_config(ck["architecture"]); m.load_state_dict(ck["model"]); m.to(dev).eval()
    n_elo = int(ck["architecture"]["n_elo_buckets"])
    ds = PackedMoveDataset(args.val_h5, sample_n=args.n, seed=0)

    def value_escore(packed_np, elo_int):
        pk = torch.from_numpy(packed_np[None]).to(dev)
        idx = elo_to_bucket(torch.tensor([elo_int]), n_elo).to(dev)
        return escore(m.forward_value(pk, elo_idx=idx))

    dv_h, dv_m, agree = [], [], []
    for i in range(len(ds)):
        row = ds[i]
        packed = row["packed_pre"].numpy().astype(np.uint8)
        elo = int(row["elo_to_move"]); opp = int(row["opp_elo"])
        board = packed_to_board(packed)
        if board.is_game_over():
            continue
        pk = torch.from_numpy(packed[None]).to(dev)
        eidx = elo_to_bucket(torch.tensor([elo]), n_elo).to(dev)
        _, squares = m.encode(pk)
        before = escore(m.value_head(m.encode(pk)[0], elo_idx=eidx))
        # model greedy two-stage move
        fl = m.from_head(squares, elo_idx=eidx)
        fmask = u64_to_mask(torch.tensor([legal_from_u64(board)], dtype=torch.int64).to(dev) if False
                            else torch.from_numpy(np.array([legal_from_u64(board)], dtype=np.uint64)).to(torch.int64))
        mf = int(fl.masked_fill(~fmask.to(fl.device), _NEG).argmax())
        tl = m.to_head(squares, torch.tensor([mf], device=dev), elo_idx=eidx)
        tmask = u64_to_mask(torch.from_numpy(np.array([legal_to_u64(board, mf)], dtype=np.uint64)).to(torch.int64)).to(tl.device)
        mt = int(tl.masked_fill(~tmask, _NEG).argmax())
        mmove = chess.Move(mf, mt)
        if mmove not in board.legal_moves:
            mmove = chess.Move(mf, mt, promotion=chess.QUEEN)
        # human move
        promo = int(row["promotion"]); hf = int(row["from_sq"]); ht = int(row["to_sq"])
        hmove = chess.Move(hf, ht, promotion=(promo if promo else None))
        if hmove not in board.legal_moves:
            continue
        bh = board.copy(); bh.push(hmove)
        bm = board.copy(); bm.push(mmove)
        dvh = (1.0 - value_escore(board_to_packed(bh), opp)) - before
        dvm = (1.0 - value_escore(board_to_packed(bm), opp)) - before
        dv_h.append(dvh); dv_m.append(dvm); agree.append(mmove == hmove)

    dv_h = np.array(dv_h); dv_m = np.array(dv_m); agree = np.array(agree)
    pct = lambda a: np.percentile(a, [1, 10, 50, 90, 99]).round(3).tolist()
    print(f"n={len(dv_h)}  agreement={agree.mean():.3f}")
    print(f"[B1] ΔV_human: mean={dv_h.mean():+.4f} (martingale~0?) std={dv_h.std():.3f} pct(1/10/50/90/99)={pct(dv_h)}")
    print(f"     ΔV_model: mean={dv_m.mean():+.4f} std={dv_m.std():.3f} pct={pct(dv_m)}")
    dis = ~agree
    d = (dv_m - dv_h)[dis]
    print(f"[B2] disagreements={dis.sum()} ({dis.mean()*100:.0f}%)  mean(ΔV_model-ΔV_human)={d.mean():+.4f}")
    print(f"     model strictly better (>+0.02): {(d>0.02).mean()*100:.0f}%  worse (<-0.02): {(d<-0.02).mean()*100:.0f}%  ~equal: {(np.abs(d)<=0.02).mean()*100:.0f}%")
    # cross-tab: at human-blunder positions (ΔV_human < -0.1), does the model pick better?
    hb = dis & (dv_h < -0.1)
    if hb.sum():
        better_at_hb = ((dv_m - dv_h)[hb] > 0).mean()
        print(f"     at human-blunder disagreements (ΔV_human<-0.1, n={hb.sum()}): model better {better_at_hb*100:.0f}% of the time")
    # model's own blunders: disagreements where model drops a lot
    mb = dis & (dv_m < -0.1)
    print(f"     model-blunder disagreements (ΔV_model<-0.1): {mb.sum()} ({mb.mean()*100:.1f}% of all positions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
