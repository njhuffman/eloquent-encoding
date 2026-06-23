"""BasePolicy: encoder + two-stage pointer heads with legality masking applied here."""
from __future__ import annotations
import torch
import torch.nn as nn
from style_policy.board_encoder import BoardEncoder
from style_policy.policy_heads import FromHead, ToHead
from style_policy.promotion_head import PromotionHead
from style_policy.legal_mask import u64_to_mask
from style_policy.packed_codec import packed_to_board_tensor
from style_policy.value_head import WDLHead

_NEG = float("-inf")


class BasePolicy(nn.Module):
    def __init__(self, encoder, from_head, to_head, promo_head, value_head):
        super().__init__()
        self.encoder = encoder
        self.from_head = from_head
        self.to_head = to_head
        self.promo_head = promo_head
        self.value_head = value_head

    @classmethod
    def from_config(cls, cfg: dict) -> "BasePolicy":
        d = int(cfg["d_model"])
        enc = BoardEncoder(d_model=d, n_layers=int(cfg["n_layers"]), nhead=int(cfg["nhead"]),
                           dim_feedforward=int(cfg["dim_feedforward"]), dropout=float(cfg["dropout"]))
        elo_dim = int(cfg.get("elo_dim", 0))
        n_elo = int(cfg.get("n_elo_buckets", 0))
        h = int(cfg["head_hidden"])
        return cls(enc,
                   FromHead(d_model=d, hidden=h, elo_dim=elo_dim, n_elo_buckets=n_elo),
                   ToHead(d_model=d, hidden=h, elo_dim=elo_dim, n_elo_buckets=n_elo),
                   PromotionHead(d_model=d),
                   WDLHead(d_model=d, hidden=h, elo_dim=elo_dim, n_elo_buckets=n_elo))

    def encode(self, packed_pre: torch.Tensor):
        board = packed_to_board_tensor(packed_pre).to(next(self.parameters()).device)
        return self.encoder(board)

    def forward_from(self, packed_pre, from_legal_u64, *, elo_idx=None):
        _, squares = self.encode(packed_pre)
        logits = self.from_head(squares, elo_idx=elo_idx)
        mask = u64_to_mask(from_legal_u64).to(logits.device)
        return logits.masked_fill(~mask, _NEG), mask

    def forward_to(self, packed_pre, from_sq, to_legal_u64, *, elo_idx=None):
        _, squares = self.encode(packed_pre)
        logits = self.to_head(squares, from_sq, elo_idx=elo_idx)
        mask = u64_to_mask(to_legal_u64).to(logits.device)
        return logits.masked_fill(~mask, _NEG), mask

    def forward_value(self, packed_pre, *, elo_idx=None):
        cls, _ = self.encode(packed_pre)
        return self.value_head(cls, elo_idx=elo_idx)

    def forward_policy(self, packed_pre, from_sq, from_legal_u64, to_legal_u64, *, elo_idx=None):
        """Encode once; return (from_logits, from_mask, to_logits, to_mask, value_logits)."""
        cls, squares = self.encode(packed_pre)
        from_logits = self.from_head(squares, elo_idx=elo_idx)
        from_mask = u64_to_mask(from_legal_u64).to(from_logits.device)
        to_logits = self.to_head(squares, from_sq, elo_idx=elo_idx)
        to_mask = u64_to_mask(to_legal_u64).to(to_logits.device)
        value_logits = self.value_head(cls, elo_idx=elo_idx)
        return (from_logits.masked_fill(~from_mask, float("-inf")), from_mask,
                to_logits.masked_fill(~to_mask, float("-inf")), to_mask, value_logits)
