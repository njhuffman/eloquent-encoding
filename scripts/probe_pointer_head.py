#!/usr/bin/env python3
"""Frozen-encoder probe: train ONE fresh head variant (factored / pointer+CLS / pointer no-CLS)
on a frozen MultiBandPolicy encoder, pooled over all bands (elo-agnostic).

Mirrors style_policy.band_head.train_band_head's structure (load ckpt, freeze encoder, fresh head,
AdamW + bf16 autocast training loop over PackedMoveDataset), but swaps in MultiBandPolicy, passes
history to encode(), and supports the pointer-head variants alongside the factored baseline.

Eval (joint move-match on a held-out set) is intentionally NOT implemented here — see Task 4.
"""
from __future__ import annotations
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from style_policy.band_head import BandHead
from style_policy.board_encode import legal_move_matrix, packed_to_board
from style_policy.dataset import PackedMoveDataset
from style_policy.legal_mask import u64_to_mask
from style_policy.loss import masked_square_ce
from style_policy.multiband_policy import MultiBandPolicy
from style_policy.pointer_head import PointerHead, joint_ce

VARIANTS = ("factored", "pointer_cls", "pointer_nocls")


def build_head(variant: str, d: int, h: int):
    if variant == "factored":
        return BandHead(d, h, use_cls=True)
    if variant == "pointer_cls":
        return PointerHead(d, d_head=64, n_heads=1, use_cls=True)
    if variant == "pointer_nocls":
        return PointerHead(d, d_head=64, n_heads=1, use_cls=False)
    raise ValueError(f"unknown variant {variant!r}")


def train_probe_head(ckpt, variant, train_h5, *, device="cuda", steps=0, batch_size=256,
                     sample_n=8_000_000, lr=3e-4, label_smoothing=0.1, num_workers=4, seed=1,
                     out=None):
    ck = torch.load(ckpt, map_location=device)
    arch = ck["architecture"]
    model = MultiBandPolicy.from_config(arch)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    d = int(arch["d_model"]); h = int(arch["head_hidden"])
    n_ply = int(arch.get("n_history_ply", 0)) if arch.get("use_last_move") else 0

    head = build_head(variant, d, h)
    head.to(device).train()
    opt = torch.optim.AdamW(head.parameters(), lr=lr)

    ds = PackedMoveDataset(train_h5, sample_n=sample_n, seed=seed)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                    collate_fn=PackedMoveDataset.collate)

    use_amp = device == "cuda"
    max_steps = steps if steps > 0 else len(dl)
    step = 0
    last_loss = None
    while step < max_steps:
        for batch in dl:
            if step >= max_steps:
                break
            packed = batch["packed_pre"].to(device)
            from_sq = batch["from_sq"].to(device); to_sq = batch["to_sq"].to(device)
            if n_ply > 0:
                hf = batch["hist_from"][:, :n_ply].to(device)
                ht = batch["hist_to"][:, :n_ply].to(device)
                hc = batch["hist_cap"][:, :n_ply].to(device)
                hist = (hf, ht, hc)
            else:
                hist = None
            with torch.no_grad():
                cls, squares = model.encode(packed, hist=hist)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                if variant == "factored":
                    fmask = u64_to_mask(batch["from_legal_u64"].to(device))
                    tmask = u64_to_mask(batch["to_legal_u64"].to(device))
                    fl = head.from_logits(squares, cls)
                    tl = head.to_logits(squares, from_sq, cls)
                    loss = (masked_square_ce(fl, from_sq, fmask, label_smoothing=label_smoothing)
                            + masked_square_ce(tl, to_sq, tmask, label_smoothing=label_smoothing))
                else:
                    pk_np = batch["packed_pre"].numpy()  # (B,34) uint8
                    b = pk_np.shape[0]
                    lm = np.zeros((b, 64, 64), dtype=bool)
                    for i in range(b):
                        lm[i] = legal_move_matrix(packed_to_board(pk_np[i]))
                    legal_mat = torch.from_numpy(lm).to(device)
                    logits = head(squares, cls if head.use_cls else None)
                    loss = joint_ce(logits, from_sq, to_sq, legal_mat, label_smoothing=label_smoothing)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            step += 1
            last_loss = float(loss.item())
            if step % 50 == 0 or step == max_steps:
                print(f"step {step}/{max_steps} loss {last_loss:.4f}")

    meta = {
        "variant": variant,
        "state_dict": head.state_dict(),
        "source_checkpoint": str(ckpt),
        "d_model": d,
        "head_hidden": h,
        "d_head": 64,
        "n_heads": 1,
        "use_cls": head.use_cls if variant != "factored" else True,
        "n_ply": n_ply,
    }
    if out is not None:
        torch.save(meta, out)
    return head, meta, last_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--train-h5", default="/mnt/eloquence_bulk/databases/wdl_history_128M.h5")
    ap.add_argument("--sample-n", type=int, default=8_000_000)
    ap.add_argument("--steps", type=int, default=0, help="0 = run one pass over the sampled data")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    _, meta, last_loss = train_probe_head(
        a.ckpt, a.variant, a.train_h5, device=a.device, steps=a.steps, batch_size=a.batch,
        sample_n=a.sample_n, lr=a.lr, label_smoothing=a.label_smoothing,
        num_workers=a.num_workers, seed=a.seed, out=a.out,
    )
    print("saved", a.out, "variant", meta["variant"], "final_loss", last_loss)


if __name__ == "__main__":
    raise SystemExit(main())
