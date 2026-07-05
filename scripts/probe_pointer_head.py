#!/usr/bin/env python3
"""Frozen-encoder probe: train fresh head variant(s) (factored / pointer+CLS / pointer no-CLS)
on a frozen MultiBandPolicy encoder, pooled over all bands (elo-agnostic).

Mirrors style_policy.band_head.train_band_head's structure (load ckpt, freeze encoder, fresh head,
AdamW + bf16 autocast training loop over PackedMoveDataset), but swaps in MultiBandPolicy, passes
history to encode(), and supports the pointer-head variants alongside the factored baseline.

Eval is joint top-1 move-match on a held-out val set (see `evaluate`): the factored head is
joint-decoded (argmax over legal (from,to) of log P(from) + log P(to|from), mirroring
style_policy.multiband_bot's joint-decoding path) rather than the greedy from-then-to decode
used elsewhere (e.g. style_policy.band_head.eval_band_head_row), so all three variants are
compared on the same metric. `--compare` trains and evaluates all three variants in one run.
"""
from __future__ import annotations
import argparse

import chess
import h5py
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
_NEG = float("-inf")
_LABELS = {"factored": "factored(fresh)", "pointer_cls": "pointer + CLS", "pointer_nocls": "pointer  no-CLS"}


def build_head(variant: str, d: int, h: int):
    if variant == "factored":
        return BandHead(d, h, use_cls=True)
    if variant == "pointer_cls":
        return PointerHead(d, d_head=64, n_heads=1, use_cls=True)
    if variant == "pointer_nocls":
        return PointerHead(d, d_head=64, n_heads=1, use_cls=False)
    raise ValueError(f"unknown variant {variant!r}")


def load_frozen_model(ckpt, device="cuda"):
    """Load + freeze the MultiBandPolicy encoder once so --compare can reuse it across variants."""
    ck = torch.load(ckpt, map_location=device)
    arch = ck["architecture"]
    model = MultiBandPolicy.from_config(arch)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, arch


def n_history_ply(arch: dict) -> int:
    return int(arch.get("n_history_ply", 0)) if arch.get("use_last_move") else 0


def train_probe_head(model, arch, variant, train_h5, *, sample_n=8_000_000, steps=0, batch=256,
                     lr=3e-4, label_smoothing=0.1, num_workers=4, seed=1, device="cuda"):
    """Train ONE fresh head variant on the (already frozen) `model` encoder. Returns the head."""
    d = int(arch["d_model"]); h = int(arch["head_hidden"])
    n_ply = n_history_ply(arch)

    head = build_head(variant, d, h)
    head.to(device).train()
    opt = torch.optim.AdamW(head.parameters(), lr=lr)

    ds = PackedMoveDataset(train_h5, sample_n=sample_n, seed=seed)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=num_workers,
                    collate_fn=PackedMoveDataset.collate)

    use_amp = device == "cuda"
    max_steps = steps if steps > 0 else len(dl)
    step = 0
    last_loss = None
    while step < max_steps:
        for b in dl:
            if step >= max_steps:
                break
            packed = b["packed_pre"].to(device)
            from_sq = b["from_sq"].to(device); to_sq = b["to_sq"].to(device)
            if n_ply > 0:
                hf = b["hist_from"][:, :n_ply].to(device)
                ht = b["hist_to"][:, :n_ply].to(device)
                hc = b["hist_cap"][:, :n_ply].to(device)
                hist = (hf, ht, hc)
            else:
                hist = None
            with torch.no_grad():
                cls, squares = model.encode(packed, hist=hist)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                if variant == "factored":
                    fmask = u64_to_mask(b["from_legal_u64"].to(device))
                    tmask = u64_to_mask(b["to_legal_u64"].to(device))
                    fl = head.from_logits(squares, cls)
                    tl = head.to_logits(squares, from_sq, cls)
                    loss = (masked_square_ce(fl, from_sq, fmask, label_smoothing=label_smoothing)
                            + masked_square_ce(tl, to_sq, tmask, label_smoothing=label_smoothing))
                else:
                    pk_np = b["packed_pre"].numpy()  # (B,34) uint8
                    bsz = pk_np.shape[0]
                    lm = np.zeros((bsz, 64, 64), dtype=bool)
                    for i in range(bsz):
                        lm[i] = legal_move_matrix(packed_to_board(pk_np[i]))
                    legal_mat = torch.from_numpy(lm).to(device)
                    logits = head(squares, cls if head.use_cls else None)
                    loss = joint_ce(logits, from_sq, to_sq, legal_mat, label_smoothing=label_smoothing)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            step += 1
            last_loss = float(loss.item())
            if step % 50 == 0 or step == max_steps:
                print(f"[{variant}] step {step}/{max_steps} loss {last_loss:.4f}")

    print(f"[{variant}] done: {step} steps, final_loss={last_loss}")
    return head


@torch.no_grad()
def evaluate(model, head, variant, val_h5, n, device, n_ply) -> float:
    """Joint top-1 move-match % on the first `n` rows of `val_h5`.

    factored: joint-decode (argmax over legal (from,to) of log P(from) + log P(to|from)),
    mirroring style_policy.multiband_bot.MultiBandBot's joint-decoding path.
    pointer_*: argmax over the legal-masked (64,64) joint logits directly.
    """
    model.eval()
    head = head.to(device).eval()
    with h5py.File(val_h5, "r") as f:
        m = min(n, f["packed_pre"].shape[0])
        packed = f["packed_pre"][:m]
        from_sq = f["from_sq"][:m]
        to_sq = f["to_sq"][:m]
        if n_ply > 0:
            hist_from = f["hist_from"][:m, :n_ply]
            hist_to = f["hist_to"][:m, :n_ply]
            hist_cap = f["hist_cap"][:m, :n_ply]

    matches = 0
    counted = 0
    for i in range(m):
        pk_np = np.asarray(packed[i], dtype=np.uint8)
        board = packed_to_board(pk_np)
        if board.is_game_over():
            continue
        counted += 1

        pk = torch.from_numpy(pk_np[None]).to(torch.int64).to(device)
        if n_ply > 0:
            hist = (
                torch.from_numpy(np.asarray(hist_from[i], dtype=np.int64)[None]).to(device),
                torch.from_numpy(np.asarray(hist_to[i], dtype=np.int64)[None]).to(device),
                torch.from_numpy(np.asarray(hist_cap[i], dtype=np.int64)[None]).to(device),
            )
        else:
            hist = None
        cls, squares = model.encode(pk, hist=hist)

        if variant == "factored":
            # joint: dedupe legal (from,to) pairs (prefer queen promo), argmax over
            # log P(from) + log P(to|from). Mirrors multiband_bot.py's `# joint:` path.
            by_ft: dict = {}
            for mv in board.legal_moves:
                k = (mv.from_square, mv.to_square)
                if k not in by_ft or mv.promotion == chess.QUEEN:
                    by_ft[k] = mv
            froms = sorted({f for f, _ in by_ft}); fidx = {f: i for i, f in enumerate(froms)}
            log_pfrom = torch.log_softmax(head.from_logits(squares, cls)[0][froms], dim=-1)
            tl = head.to_logits(squares.expand(len(froms), -1, -1),
                                torch.tensor(froms, device=device), cls.expand(len(froms), -1))
            tos_by_from: dict = {}
            for (f, t) in by_ft:
                tos_by_from.setdefault(f, []).append(t)
            best = None
            for f in froms:
                tos = tos_by_from[f]
                log_pto = torch.log_softmax(tl[fidx[f]][tos], dim=-1)
                for ti, t in enumerate(tos):
                    score = float(log_pfrom[fidx[f]]) + float(log_pto[ti])
                    if best is None or score > best[0]:
                        best = (score, f, t)
            _, pf, pt = best
        else:
            legal = torch.from_numpy(legal_move_matrix(board)).to(device)
            logits = head(squares, cls if head.use_cls else None)  # (1,64,64)
            idx = int(logits.masked_fill(~legal, _NEG).view(-1).argmax())
            pf, pt = idx // 64, idx % 64

        if pf == int(from_sq[i]) and pt == int(to_sq[i]):
            matches += 1

    return 100.0 * matches / counted if counted else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", choices=VARIANTS, help="single-variant mode (ignored with --compare)")
    ap.add_argument("--compare", action="store_true", help="train+eval all 3 variants and print a table")
    ap.add_argument("--train-h5", default="/mnt/eloquence_bulk/databases/wdl_history_128M.h5")
    ap.add_argument("--val-h5", default="/mnt/eloquence_bulk/databases/wdl_validation_2025_05.h5")
    ap.add_argument("--eval-n", type=int, default=20_000, help="0 disables eval")
    ap.add_argument("--sample-n", type=int, default=8_000_000)
    ap.add_argument("--steps", type=int, default=0, help="0 = run one pass over the sampled data")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", help="required in single-variant mode")
    a = ap.parse_args()

    model, arch = load_frozen_model(a.ckpt, device=a.device)
    n_ply = n_history_ply(arch)

    def train(variant):
        return train_probe_head(
            model, arch, variant, a.train_h5, sample_n=a.sample_n, steps=a.steps, batch=a.batch,
            lr=a.lr, label_smoothing=a.label_smoothing, num_workers=a.num_workers, seed=a.seed,
            device=a.device,
        )

    if a.compare:
        valname = a.val_h5.split("/")[-1]
        rows = []
        for variant in VARIANTS:
            head = train(variant)
            mm = evaluate(model, head, variant, a.val_h5, a.eval_n, a.device, n_ply) if a.eval_n > 0 else float("nan")
            steps_taken = a.steps if a.steps > 0 else "1ep"
            rows.append((_LABELS[variant], steps_taken, mm))
        print(f"\n{'variant':<18}{'steps':>7}   move%_{valname}")
        for label, steps_taken, mm in rows:
            print(f"{label:<18}{str(steps_taken):>7}   {mm:.2f}")
        return 0

    if not a.variant or not a.out:
        raise SystemExit("--variant and --out are required unless --compare is set")

    head = train(a.variant)
    d = int(arch["d_model"]); h = int(arch["head_hidden"])
    meta = {
        "variant": a.variant,
        "state_dict": head.state_dict(),
        "source_checkpoint": str(a.ckpt),
        "d_model": d,
        "head_hidden": h,
        "d_head": 64,
        "n_heads": 1,
        "use_cls": head.use_cls if a.variant != "factored" else True,
        "n_ply": n_ply,
    }
    torch.save(meta, a.out)
    print("saved", a.out, "variant", a.variant)
    if a.eval_n > 0:
        mm = evaluate(model, head, a.variant, a.val_h5, a.eval_n, a.device, n_ply)
        print(f"move%_{a.val_h5.split('/')[-1]} = {mm:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
