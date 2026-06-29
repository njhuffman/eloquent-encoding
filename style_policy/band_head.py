"""Per-band specialized heads on a frozen encoder (elo-agnostic conditioning by hard band split)."""
from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import h5py
from style_policy.policy_heads import FromHead, ToHead
from style_policy.model import BasePolicy
from style_policy.dataset import PackedMoveDataset
from style_policy.loss import masked_square_ce
from style_policy.legal_mask import u64_to_mask
from style_policy.board_encode import packed_to_board, legal_from_u64, legal_to_u64
from style_policy.model_spec import elo_to_bucket

class BandHead(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.from_head = FromHead(d_model=d_model, hidden=hidden, elo_dim=0)
        self.to_head = ToHead(d_model=d_model, hidden=hidden, elo_dim=0)

    def from_logits(self, squares: torch.Tensor) -> torch.Tensor:
        return self.from_head(squares)

    def to_logits(self, squares: torch.Tensor, from_sq: torch.Tensor) -> torch.Tensor:
        return self.to_head(squares, from_sq)


def train_band_head(checkpoint, band, train_h5, *, device="cuda", steps=2000,
                    batch_size=256, sample_n=None, lr=3e-4, label_smoothing=0.0,
                    num_workers=4, seed=1, out=None):
    ck = torch.load(checkpoint, map_location=device)
    arch = ck["architecture"]
    # strict=False: policy-only checkpoints (e.g. base_64M, trained on j3 data) have no value
    # head; the band-head path uses only the encoder + from/to heads, so a missing value head is fine.
    model = BasePolicy.from_config(arch); model.load_state_dict(ck["model"], strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d, h = int(arch["d_model"]), int(arch["head_hidden"])
    head = BandHead(d, h).to(device); head.train()
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    ds = PackedMoveDataset(train_h5, sample_n=sample_n, seed=seed, band=(band, band + 100))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                    collate_fn=PackedMoveDataset.collate)
    use_amp = device == "cuda"
    step = 0
    while step < steps:
        for batch in dl:
            if step >= steps:
                break
            packed = batch["packed_pre"].to(device)
            from_sq = batch["from_sq"].to(device); to_sq = batch["to_sq"].to(device)
            fmask = u64_to_mask(batch["from_legal_u64"].to(device))
            tmask = u64_to_mask(batch["to_legal_u64"].to(device))
            with torch.no_grad():
                _, squares = model.encode(packed)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                fl = head.from_logits(squares)
                tl = head.to_logits(squares, from_sq)
                loss = (masked_square_ce(fl, from_sq, fmask, label_smoothing=label_smoothing)
                        + masked_square_ce(tl, to_sq, tmask, label_smoothing=label_smoothing))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            step += 1
    meta = {"d_model": d, "hidden": h, "source_checkpoint": str(checkpoint), "band": int(band)}
    if out is not None:
        torch.save({"band_head": head.state_dict(), **meta}, out)
    return head, meta


_NEG = float("-inf")

def _mask1(u64, dev):
    return u64_to_mask(torch.from_numpy(np.array([u64], dtype=np.uint64)).to(torch.int64)).to(dev)

@torch.no_grad()
def eval_band_head_row(checkpoint, band_head, val_h5, bands, *, device="cuda", n=10000):
    ck = torch.load(checkpoint, map_location=device)
    arch = ck["architecture"]; n_elo = int(arch["n_elo_buckets"])
    model = BasePolicy.from_config(arch); model.load_state_dict(ck["model"], strict=False)  # see train_band_head
    model.to(device).eval()
    band_head = band_head.to(device).eval()
    with h5py.File(val_h5, "r") as f:
        m = min(n, f["packed_pre"].shape[0])
        packed = f["packed_pre"][:m]; hf = f["from_sq"][:m]; ht = f["to_sq"][:m]; elo = f["elo_to_move"][:m]
    out = {b: {"spec": 0, "shared": 0, "count": 0} for b in bands}
    for i in range(m):
        b = int(min(max(bands), max(min(bands), (int(elo[i]) // 100) * 100)))
        if b not in out:
            continue
        board = packed_to_board(np.asarray(packed[i], np.uint8))
        if board.is_game_over():
            continue
        out[b]["count"] += 1
        pk = torch.from_numpy(np.asarray(packed[i], np.uint8)[None]).to(device)
        _, squares = model.encode(pk)
        fmask = _mask1(legal_from_u64(board), device)
        bi = elo_to_bucket(torch.tensor([b]), n_elo).to(device)
        for tag, ffn, tfn in (
            ("spec", lambda s: band_head.from_logits(s), lambda s, pf: band_head.to_logits(s, pf)),
            ("shared", lambda s: model.from_head(s, elo_idx=bi), lambda s, pf: model.to_head(s, pf, elo_idx=bi)),
        ):
            pf = int(ffn(squares).masked_fill(~fmask, _NEG).argmax())
            tmask = _mask1(legal_to_u64(board, pf), device)
            pft = torch.tensor([pf], device=device)
            pt = int(tfn(squares, pft).masked_fill(~tmask, _NEG).argmax())
            if pf == int(hf[i]) and pt == int(ht[i]):
                out[b][tag] += 1
    return {b: {"spec": 100.0 * v["spec"] / v["count"] if v["count"] else 0.0,
                "shared": 100.0 * v["shared"] / v["count"] if v["count"] else 0.0,
                "count": v["count"]} for b, v in out.items()}
