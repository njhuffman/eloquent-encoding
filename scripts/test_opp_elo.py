#!/usr/bin/env python3
"""Does opp_elo carry signal? Fixed self-band, frozen encoder: train an unconditioned head vs an
opp_elo-conditioned head on the SAME data/features (shared encode), compare val CE + move-match.
If the opp_elo head isn't meaningfully better, opp_elo isn't worth re-introducing conditioning for.
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import h5py
from torch.utils.data import DataLoader
from style_policy.model import BasePolicy
from style_policy.policy_heads import FromHead, ToHead
from style_policy.dataset import PackedMoveDataset
from style_policy.loss import masked_square_ce
from style_policy.legal_mask import u64_to_mask
from style_policy.model_spec import elo_to_bucket
from style_policy.board_encode import packed_to_board, legal_from_u64, legal_to_u64

_NEG = float("-inf")


def _m1(u64, dev):
    return u64_to_mask(torch.from_numpy(np.array([u64], dtype=np.uint64)).to(torch.int64)).to(dev)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="style_policy_checkpoints/base_64M/base_64M_stage_1.pt")
    ap.add_argument("--band", type=int, default=1500)
    ap.add_argument("--train-h5", default="/mnt/eloquence_bulk/databases/wdl_training_16M.h5")
    ap.add_argument("--val-h5", default="/mnt/eloquence_bulk/databases/wdl_validation_1M.h5")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--elo-dim", type=int, default=32)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    ck = torch.load(a.checkpoint, map_location=dev)
    arch = ck["architecture"]; n_elo = int(arch["n_elo_buckets"])
    d = int(arch["d_model"]); h = int(arch["head_hidden"])
    model = BasePolicy.from_config(arch); model.load_state_dict(ck["model"], strict=False)
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    torch.manual_seed(0)
    base_f = FromHead(d_model=d, hidden=h, elo_dim=0).to(dev)
    base_t = ToHead(d_model=d, hidden=h, elo_dim=0).to(dev)
    opp_f = FromHead(d_model=d, hidden=h, elo_dim=a.elo_dim, n_elo_buckets=n_elo).to(dev)
    opp_t = ToHead(d_model=d, hidden=h, elo_dim=a.elo_dim, n_elo_buckets=n_elo).to(dev)
    opt_base = torch.optim.AdamW(list(base_f.parameters()) + list(base_t.parameters()), lr=3e-4)
    opt_opp = torch.optim.AdamW(list(opp_f.parameters()) + list(opp_t.parameters()), lr=3e-4)

    ds = PackedMoveDataset(a.train_h5, seed=1, band=(a.band, a.band + 100))
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=4, collate_fn=PackedMoveDataset.collate)
    print(f"band {a.band}: {len(ds):,} train samples; training base vs opp_elo heads ({a.steps} steps)", flush=True)
    for g in (base_f, base_t, opp_f, opp_t):
        g.train()
    step = 0
    while step < a.steps:
        for b in dl:
            if step >= a.steps:
                break
            packed = b["packed_pre"].to(dev); fs = b["from_sq"].to(dev); ts = b["to_sq"].to(dev)
            fm = u64_to_mask(b["from_legal_u64"].to(dev)); tm = u64_to_mask(b["to_legal_u64"].to(dev))
            oi = elo_to_bucket(b["opp_elo"], n_elo).to(dev)
            with torch.no_grad():
                _, sq = model.encode(packed)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                lb = masked_square_ce(base_f(sq), fs, fm, label_smoothing=a.ls) + masked_square_ce(base_t(sq, fs), ts, tm, label_smoothing=a.ls)
            opt_base.zero_grad(set_to_none=True); lb.backward(); opt_base.step()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                lo = masked_square_ce(opp_f(sq, elo_idx=oi), fs, fm, label_smoothing=a.ls) + masked_square_ce(opp_t(sq, fs, elo_idx=oi), ts, tm, label_smoothing=a.ls)
            opt_opp.zero_grad(set_to_none=True); lo.backward(); opt_opp.step()
            step += 1
    for g in (base_f, base_t, opp_f, opp_t):
        g.eval()
    torch.set_grad_enabled(False)

    # ---- eval on the band's val positions ----
    with h5py.File(a.val_h5, "r") as f:
        elo = f["elo_to_move"][:]
        idx = np.nonzero((elo >= a.band) & (elo < a.band + 100))[0]
        packed = f["packed_pre"][:][idx]; hf = f["from_sq"][:][idx]; ht = f["to_sq"][:][idx]
        flu = f["from_legal_u64"][:][idx]; tlu = f["to_legal_u64"][:][idx]; oe = f["opp_elo"][:][idx]
    m = len(idx)
    ce_b = ce_o = 0.0; nb = 0
    top_b = top_o = cnt = 0
    for i in range(m):
        pk = torch.from_numpy(np.asarray(packed[i], np.uint8)[None]).to(dev)
        _, sq = model.encode(pk)
        fm = _m1(int(flu[i]), dev); tm = _m1(int(tlu[i]), dev)
        fs = torch.tensor([int(hf[i])], device=dev); ts = torch.tensor([int(ht[i])], device=dev)
        oi = elo_to_bucket(torch.tensor([int(oe[i])]), n_elo).to(dev)
        # teacher-forced CE (true from_sq, stored masks -> correct)
        ce_b += float(masked_square_ce(base_f(sq), fs, fm) + masked_square_ce(base_t(sq, fs), ts, tm))
        ce_o += float(masked_square_ce(opp_f(sq, elo_idx=oi), fs, fm) + masked_square_ce(opp_t(sq, fs, elo_idx=oi), ts, tm))
        nb += 1
        # free-running top-1 (need board for predicted-from to-mask)
        board = packed_to_board(np.asarray(packed[i], np.uint8))
        if board.is_game_over():
            continue
        cnt += 1
        fmask = _m1(legal_from_u64(board), dev)
        for tag, ff, tf in (("b", base_f, base_t), ("o", opp_f, opp_t)):
            if tag == "b":
                pf = int(ff(sq).masked_fill(~fmask, _NEG).argmax())
                pt = int(tf(sq, torch.tensor([pf], device=dev)).masked_fill(~_m1(legal_to_u64(board, pf), dev), _NEG).argmax())
            else:
                pf = int(ff(sq, elo_idx=oi).masked_fill(~fmask, _NEG).argmax())
                pt = int(tf(sq, torch.tensor([pf], device=dev), elo_idx=oi).masked_fill(~_m1(legal_to_u64(board, pf), dev), _NEG).argmax())
            if pf == int(hf[i]) and pt == int(ht[i]):
                if tag == "b": top_b += 1
                else: top_o += 1

    print(f"\nval band {a.band}  (n={m}, eval positions {cnt})")
    print(f"  from+to CE     base={ce_b/nb:.4f}   opp_elo={ce_o/nb:.4f}   delta={ce_o/nb - ce_b/nb:+.4f} (negative = opp_elo better)")
    print(f"  move-match top1 base={100*top_b/cnt:.2f}%  opp_elo={100*top_o/cnt:.2f}%  delta={100*(top_o-top_b)/cnt:+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
