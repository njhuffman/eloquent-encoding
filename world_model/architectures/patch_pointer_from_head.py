"""From-square logits via CLS-conditioned query dotted with patch tokens (learned Q/K)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from jepa3.architectures.chess_jepa_v3 import _mlp_gelu


class PatchPointerFromHead(nn.Module):
    """64 logits: dot(W_q(trunk([CLS; phi(ELO)])), W_k(patch_i)) * scale."""

    def __init__(
        self,
        *,
        d_model: int,
        pointer_dim: int,
        query_hidden: int,
        query_depth: int,
        elo_dim: int,
    ) -> None:
        super().__init__()
        d = int(d_model)
        h = int(pointer_dim)
        if h < 1:
            raise ValueError(f"pointer_dim must be >= 1 (got {pointer_dim})")
        self.d_model = d
        self.pointer_dim = h
        elo_dim = int(elo_dim)
        self.elo_dim = elo_dim
        if elo_dim > 0:
            self.elo_proj = nn.Linear(1, elo_dim)
            trunk_in = d + elo_dim
        else:
            self.elo_proj = None
            trunk_in = d
        self.query_trunk = _mlp_gelu(trunk_in, int(query_hidden), int(query_depth), d)
        self.query_proj = nn.Linear(d, h)
        self.key_proj = nn.Linear(d, h)
        self.register_buffer("_scale", torch.tensor(1.0 / math.sqrt(float(h))), persistent=False)

    def forward(
        self,
        z_global: torch.Tensor,
        patch_tokens: torch.Tensor,
        elo: torch.Tensor | None,
    ) -> torch.Tensor:
        """``z_global`` (B, D), ``patch_tokens`` (B, 64, D); ``elo`` (B,) required if elo_dim>0 else ignored."""
        if z_global.ndim != 2 or z_global.shape[-1] != self.d_model:
            raise ValueError(f"z_global must be (B, {self.d_model}), got {tuple(z_global.shape)}")
        if patch_tokens.ndim != 3 or patch_tokens.shape[1] != 64 or patch_tokens.shape[-1] != self.d_model:
            raise ValueError(
                f"patch_tokens must be (B, 64, {self.d_model}), got {tuple(patch_tokens.shape)}"
            )
        b = z_global.shape[0]
        if patch_tokens.shape[0] != b:
            raise ValueError(
                f"z_global batch {b} != patch_tokens batch {patch_tokens.shape[0]}"
            )
        if self.elo_proj is not None:
            if elo is None:
                raise ValueError("elo tensor required when from_pointer_elo_dim > 0")
            if elo.ndim != 1 or elo.shape[0] != b:
                raise ValueError(f"elo must be (B,) with B={b}, got {tuple(elo.shape)}")
            e = elo.to(dtype=z_global.dtype, device=z_global.device).unsqueeze(-1)
            elo_feat = self.elo_proj(e)
            u = torch.cat([z_global, elo_feat], dim=-1)
        else:
            u = z_global
        h_mid = self.query_trunk(u)
        q = self.query_proj(h_mid)
        k = self.key_proj(patch_tokens)
        logits = (q.unsqueeze(1) * k).sum(dim=-1) * self._scale.to(dtype=q.dtype)
        return logits
