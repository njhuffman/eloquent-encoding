"""Transformer predictor: four action tokens + 64 patch tokens → predicted patch reps."""

from __future__ import annotations

import torch
import torch.nn as nn

from jepa3.board_square_categories import NUM_SQUARE_CATEGORIES


class JepaPatchTransformerPredictor(nn.Module):
    """Self-attention stack on ``[from,to,moved,placed] + patch_tokens`` → last 64 outputs."""

    def __init__(
        self,
        *,
        d_model: int,
        square_embed_dim: int,
        n_layers: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = int(d_model)
        self.slot_proj = nn.Linear(int(square_embed_dim), self.d_model)
        self.moved_cat_emb = nn.Embedding(NUM_SQUARE_CATEGORIES, self.d_model)
        self.placed_cat_emb = nn.Embedding(NUM_SQUARE_CATEGORIES, self.d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(n_layers))
        nn.init.trunc_normal_(self.moved_cat_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.placed_cat_emb.weight, std=0.02)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        e_from: torch.Tensor,
        e_to: torch.Tensor,
        moved_cat: torch.Tensor,
        placed_cat: torch.Tensor,
    ) -> torch.Tensor:
        """patch_tokens (B,64,D); e_from/e_to (B, square_embed_dim); cats (B,) long → (B,64,D)."""
        if patch_tokens.ndim != 3 or patch_tokens.shape[1] != 64:
            raise ValueError(f"patch_tokens must be (B, 64, d_model), got {tuple(patch_tokens.shape)}")
        if patch_tokens.shape[-1] != self.d_model:
            raise ValueError(f"patch_tokens last dim must be d_model={self.d_model}")
        b = patch_tokens.shape[0]
        t0 = self.slot_proj(e_from).unsqueeze(1)
        t1 = self.slot_proj(e_to).unsqueeze(1)
        t2 = self.moved_cat_emb(moved_cat.long()).unsqueeze(1)
        t3 = self.placed_cat_emb(placed_cat.long()).unsqueeze(1)
        x = torch.cat([t0, t1, t2, t3, patch_tokens], dim=1)
        assert x.shape == (b, 68, self.d_model)
        out = self.encoder(x)
        return out[:, -64:, :]
