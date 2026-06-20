"""MLP-Mixer on delta-z history + residual MLP; frozen GFP from-square head on z."""

from __future__ import annotations

import torch
import torch.nn as nn

from gfp.model import FromSquareMlpHead


class Mlp(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = int(in_dim)
        h = int(hidden)
        for _ in range(int(depth) - 1):
            layers.append(nn.Linear(d, h))
            layers.append(nn.GELU())
            d = h
        layers.append(nn.Linear(d, int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MixerBlock(nn.Module):
    """One mixer block: token-mixing along sequence then channel-mixing (Tolstikhin-style)."""

    def __init__(
        self,
        seq_len: int,
        channels: int,
        tokens_mlp_dim: int,
        channels_mlp_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.token_mix = Mlp(seq_len, tokens_mlp_dim, seq_len, depth=2)
        self.norm2 = nn.LayerNorm(channels)
        self.channel_mix = Mlp(channels, channels_mlp_dim, channels, depth=2)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        y = self.norm1(x)
        y = y.transpose(1, 2)
        y = self.token_mix(y)
        y = y.transpose(1, 2)
        x = x + self.dropout(y)
        y = self.norm2(x)
        x = x + self.dropout(self.channel_mix(y))
        return x


class DeltaZMixer(nn.Module):
    """MLP-Mixer stack over (B, N, d_model); pooled to mixer_dim."""

    def __init__(
        self,
        *,
        history_len: int,
        d_model: int,
        mixer_dim: int,
        depth: int,
        tokens_mlp_dim: int,
        channels_mlp_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        n = int(history_len)
        d = int(d_model)
        self.blocks = nn.ModuleList(
            [
                MixerBlock(
                    seq_len=n,
                    channels=d,
                    tokens_mlp_dim=int(tokens_mlp_dim),
                    channels_mlp_dim=int(channels_mlp_dim),
                    dropout=float(dropout),
                )
                for _ in range(int(depth))
            ]
        )
        self.norm_f = nn.LayerNorm(d)
        self.proj = nn.Linear(d, int(mixer_dim))

    def forward(self, delta_z: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        """
        delta_z: (B, N, D), history_mask: (B, N) float/bool — 1 valid timestep.
        """
        x = delta_z
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        x = self.proj(x)
        m = history_mask.float().clamp(0.0, 1.0)
        denom = m.sum(dim=-1, keepdim=True).clamp(min=1.0)
        pooled = (x * m.unsqueeze(-1)).sum(dim=1) / denom
        return pooled


class ResidualFromPredictor(nn.Module):
    """
    Frozen FromSquareMlpHead(z_curr) + trainable mixer + residual MLP.
    Forward returns total logits (gfp + residual) for loss; gfp logits available for logging.
    """

    def __init__(
        self,
        gfp_head: FromSquareMlpHead,
        *,
        history_len: int,
        d_model: int,
        mixer_dim: int,
        mixer_depth: int,
        mixer_tokens_mlp_dim: int,
        mixer_channels_mlp_dim: int,
        mixer_dropout: float,
        elo_num_buckets: int,
        elo_embed_dim: int,
        residual_hidden: int,
        residual_depth: int,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.history_len = int(history_len)
        self.elo_num_buckets = int(elo_num_buckets)
        self.elo_embed_dim = int(elo_embed_dim)
        self.null_elo_index = int(elo_num_buckets)

        self.gfp_head = gfp_head
        for p in self.gfp_head.parameters():
            p.requires_grad_(False)

        self.mixer = DeltaZMixer(
            history_len=history_len,
            d_model=d_model,
            mixer_dim=mixer_dim,
            depth=mixer_depth,
            tokens_mlp_dim=mixer_tokens_mlp_dim,
            channels_mlp_dim=mixer_channels_mlp_dim,
            dropout=mixer_dropout,
        )
        self.elo_embed = nn.Embedding(self.null_elo_index + 1, self.elo_embed_dim)

        in_res = int(mixer_dim) + self.d_model + self.elo_embed_dim + 64
        self.residual_mlp = Mlp(
            in_res,
            int(residual_hidden),
            64,
            depth=int(residual_depth),
        )

    def train(self, mode: bool = True) -> ResidualFromPredictor:
        super().train(mode)
        self.gfp_head.eval()
        return self

    def trainable_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        params.extend(self.mixer.parameters())
        params.extend(self.elo_embed.parameters())
        params.extend(self.residual_mlp.parameters())
        return params

    def elo_indices_from_buckets(self, elo_bucket: torch.Tensor) -> torch.Tensor:
        """
        elo_bucket: int64 (B,) with values -1 = missing, else floor(elo/100) style bucket.
        Maps to 0..elo_num_buckets-1 or null index.
        """
        b = elo_bucket.long()
        valid = b >= 0
        clamped = b.clamp(min=0, max=self.elo_num_buckets - 1)
        idx = torch.where(valid, clamped, torch.full_like(b, self.null_elo_index))
        return idx

    def forward(
        self,
        delta_z: torch.Tensor,
        z_curr: torch.Tensor,
        history_mask: torch.Tensor,
        elo_bucket: torch.Tensor,
        *,
        elo_null_prob: float,
        train: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (total_logits, gfp_logits) both (B, 64).
        """
        z = z_curr.float()
        with torch.no_grad():
            gfp_logits = self.gfp_head(z)

        mixer_out = self.mixer(delta_z.float(), history_mask)

        elo_idx = self.elo_indices_from_buckets(elo_bucket)
        if train and elo_null_prob > 0.0:
            rng = torch.rand(elo_idx.shape[0], device=elo_idx.device, dtype=z.dtype)
            drop = rng < float(elo_null_prob)
            elo_idx = torch.where(drop, torch.full_like(elo_idx, self.null_elo_index), elo_idx)
        elo_e = self.elo_embed(elo_idx.long())

        inp = torch.cat([mixer_out, z, elo_e, gfp_logits.float()], dim=-1)
        residual = self.residual_mlp(inp)
        total = gfp_logits + residual
        return total, gfp_logits


def load_gfp_head_from_checkpoint(
    ckpt_path: str,
    *,
    d_model: int,
    device: torch.device,
) -> FromSquareMlpHead:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "head_state_dict" not in ck:
        raise KeyError(f"missing head_state_dict in {ckpt_path}")
    arch = ck.get("architecture") or {}
    cfg = arch.get("config") if isinstance(arch, dict) else {}
    if not isinstance(cfg, dict):
        cfg = {}
    hidden = int(cfg.get("head_hidden", 512))
    depth = int(cfg.get("head_depth", 2))
    head = FromSquareMlpHead(int(d_model), hidden, depth).to(device)
    head.load_state_dict(ck["head_state_dict"], strict=True)
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head
