#!/usr/bin/env python3
"""Does feeding the CLS token to the policy heads help move prediction?

Frozen encoder (default the value-supervised multiband_64M encoder, whose CLS is rich), fixed
self-band, shared-encode A/B: baseline head (squares only) vs CLS head (squares + broadcast CLS).
Positive result => pursue feeding CLS to heads; negative is inconclusive (per user's framing).
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn as nn
import h5py
from torch.utils.data import DataLoader
from style_policy.model import BasePolicy
from style_policy.policy_heads import FromHead, ToHead
from style_policy.dataset import PackedMoveDataset
from style_policy.loss import masked_square_ce
from style_policy.legal_mask import u64_to_mask
from style_policy.board_encode import packed_to_board, legal_from_u64, legal_to_u64

_NEG = float("-inf")


def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class CLSFromHead(nn.Module):
    def __init__(self, d, hidden):
        super().__init__(); self.score = _mlp(2 * d, hidden, 1)

    def forward(self, squares, cls):
        c = cls.unsqueeze(1).expand(-1, 64, -1)
        return self.score(torch.cat([squares, c], -1)).squeeze(-1)


class CLSToHead(nn.Module):
    def __init__(self, d, hidden):
        super().__init__(); self.score = _mlp(3 * d, hidden, 1)

    def forward(self, squares, from_sq, cls):
        b, _, d = squares.shape
        origin = squares[torch.arange(b, device=squares.device), from_sq.long()].unsqueeze(1).expand(b, 64, d)
        c = cls.unsqueeze(1).expand(b, 64, d)
        return self.score(torch.cat([squares, origin, c], -1)).squeeze(-1)


def _m1(u64, dev):
    return u64_to_mask(torch.from_numpy(np.array([u64], dtype=np.uint64)).to(torch.int64)).to(dev)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="style_policy_checkpoints/multiband_64M/multiband_64M_encoder.pt")
    ap.add_argument("--band", type=int, default=1500)
    ap.add_argument("--train-h5", default="/mnt/eloquence_bulk/databases/wdl_training_16M.h5")
    ap.add_argument("--val-h5", default="/mnt/eloquence_bulk/databases/wdl_validation_1M.h5")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    ck = torch.load(a.checkpoint, map_location=dev)
    arch = ck["architecture"]; d = int(arch["d_model"]); h = int(arch["head_hidden"])
    model = BasePolicy.from_config(arch); model.load_state_dict(ck["model"], strict=False)
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    torch.manual_seed(0)
    base_f = FromHead(d_model=d, hidden=h, elo_dim=0).to(dev)
    base_t = ToHead(d_model=d, hidden=h, elo_dim=0).to(dev)
    cls_f = CLSFromHead(d, h).to(dev); cls_t = CLSToHead(d, h).to(dev)
    opt_b = torch.optim.AdamW(list(base_f.parameters()) + list(base_t.parameters()), lr=3e-4)
    opt_c = torch.optim.AdamW(list(cls_f.parameters()) + list(cls_t.parameters()), lr=3e-4)

    ds = PackedMoveDataset(a.train_h5, seed=1, band=(a.band, a.band + 100))
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=4, collate_fn=PackedMoveDataset.collate)
    print(f"band {a.band}: {len(ds):,} train samples; baseline vs CLS head, encoder={a.checkpoint.split('/')[-1]} ({a.steps} steps)", flush=True)
    for g in (base_f, base_t, cls_f, cls_t):
        g.train()
    step = 0
    while step < a.steps:
        for b in dl:
            if step >= a.steps:
                break
            packed = b["packed_pre"].to(dev); fs = b["from_sq"].to(dev); ts = b["to_sq"].to(dev)
            fm = u64_to_mask(b["from_legal_u64"].to(dev)); tm = u64_to_mask(b["to_legal_u64"].to(dev))
            with torch.no_grad():
                cls, sq = model.encode(packed)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                lb = masked_square_ce(base_f(sq), fs, fm, label_smoothing=a.ls) + masked_square_ce(base_t(sq, fs), ts, tm, label_smoothing=a.ls)
            opt_b.zero_grad(set_to_none=True); lb.backward(); opt_b.step()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                lc = masked_square_ce(cls_f(sq, cls), fs, fm, label_smoothing=a.ls) + masked_square_ce(cls_t(sq, fs, cls), ts, tm, label_smoothing=a.ls)
            opt_c.zero_grad(set_to_none=True); lc.backward(); opt_c.step()
            step += 1
    for g in (base_f, base_t, cls_f, cls_t):
        g.eval()
    torch.set_grad_enabled(False)

    with h5py.File(a.val_h5, "r") as f:
        elo = f["elo_to_move"][:]
        idx = np.nonzero((elo >= a.band) & (elo < a.band + 100))[0]
        packed = f["packed_pre"][:][idx]; hf = f["from_sq"][:][idx]; ht = f["to_sq"][:][idx]
        flu = f["from_legal_u64"][:][idx]; tlu = f["to_legal_u64"][:][idx]
    m = len(idx)
    ce_b = ce_c = 0.0; nb = 0; top_b = top_c = cnt = 0
    for i in range(m):
        pk = torch.from_numpy(np.asarray(packed[i], np.uint8)[None]).to(dev)
        cls, sq = model.encode(pk)
        fm = _m1(int(flu[i]), dev); tm = _m1(int(tlu[i]), dev)
        fs = torch.tensor([int(hf[i])], device=dev); ts = torch.tensor([int(ht[i])], device=dev)
        ce_b += float(masked_square_ce(base_f(sq), fs, fm) + masked_square_ce(base_t(sq, fs), ts, tm))
        ce_c += float(masked_square_ce(cls_f(sq, cls), fs, fm) + masked_square_ce(cls_t(sq, fs, cls), ts, tm))
        nb += 1
        board = packed_to_board(np.asarray(packed[i], np.uint8))
        if board.is_game_over():
            continue
        cnt += 1
        fmask = _m1(legal_from_u64(board), dev)
        pfb = int(base_f(sq).masked_fill(~fmask, _NEG).argmax())
        ptb = int(base_t(sq, torch.tensor([pfb], device=dev)).masked_fill(~_m1(legal_to_u64(board, pfb), dev), _NEG).argmax())
        top_b += int(pfb == int(hf[i]) and ptb == int(ht[i]))
        pfc = int(cls_f(sq, cls).masked_fill(~fmask, _NEG).argmax())
        ptc = int(cls_t(sq, torch.tensor([pfc], device=dev), cls).masked_fill(~_m1(legal_to_u64(board, pfc), dev), _NEG).argmax())
        top_c += int(pfc == int(hf[i]) and ptc == int(ht[i]))

    print(f"\nval band {a.band}  (n={m}, eval positions {cnt})")
    print(f"  from+to CE     base={ce_b/nb:.4f}   +CLS={ce_c/nb:.4f}   delta={ce_c/nb - ce_b/nb:+.4f} (negative = CLS better)")
    print(f"  move-match top1 base={100*top_b/cnt:.2f}%  +CLS={100*top_c/cnt:.2f}%  delta={100*(top_c-top_b)/cnt:+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
