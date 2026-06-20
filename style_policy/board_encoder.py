"""Transformer board encoder: 64 square tokens + CLS (+ side-to-move). Returns (cls, square_tokens).

Square token s = piece_category_embed(category[s]) + square_position_embed(s).
CLS carries side-to-move. Board-global state (castling, en-passant) is read from the board
tensor's planes; if the lifted codec exposes those planes, add dedicated special tokens here.
Heads point into the 64 SQUARE tokens (not CLS) — see policy_heads.py.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from style_policy.square_categories import NUM_SQUARE_CATEGORIES, square_categories_from_board_tensor


class BoardEncoder(nn.Module):
    def __init__(self, *, d_model: int, n_layers: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = int(d_model)
        self.piece_emb = nn.Embedding(NUM_SQUARE_CATEGORIES, d_model)
        self.square_emb = nn.Embedding(64, d_model)
        self.turn_cls_emb = nn.Embedding(2, d_model)  # index 0 = black to move, 1 = white
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        nn.init.trunc_normal_(self.square_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.turn_cls_emb.weight, std=0.02)

    def _turn_index(self, board_tensor: torch.Tensor) -> torch.Tensor:
        # Side-to-move plane: index 12 per the codec channel map (1.0 = white to move).
        # board_tensor is (B, H, W, C); reduce to (B,) long index.
        plane = board_tensor[..., 12]
        return (plane.reshape(plane.shape[0], -1).mean(dim=1) > 0.5).long()

    def forward(self, board_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cats = square_categories_from_board_tensor(board_tensor)  # (B,64)
        b = cats.shape[0]
        sq_idx = torch.arange(64, device=cats.device).unsqueeze(0).expand(b, 64)
        tok = self.piece_emb(cats) + self.square_emb(sq_idx)  # (B,64,d)
        turn = self.turn_cls_emb(self._turn_index(board_tensor)).unsqueeze(1)  # (B,1,d)
        x = torch.cat([turn, tok], dim=1)  # (B,65,d)
        h = self.encoder(x)
        return h[:, 0], h[:, 1:]  # cls, square_tokens
