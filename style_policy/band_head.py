"""Per-band specialized heads on a frozen encoder (elo-agnostic conditioning by hard band split)."""
from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from style_policy.policy_heads import FromHead, ToHead
from style_policy.model import BasePolicy
from style_policy.dataset import PackedMoveDataset
from style_policy.loss import masked_square_ce
from style_policy.legal_mask import u64_to_mask

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
    model = BasePolicy.from_config(arch); model.load_state_dict(ck["model"])
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
