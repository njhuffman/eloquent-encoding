"""
Chess-JEPA v1: spatial-token encoder, EMA target twin, Elo-conditioned predictor.
"""

from __future__ import annotations

import copy
import warnings
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa.config import BOARD_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH

ARCHITECTURE_ID = "chess_jepa_v1"

DEFAULT_ARCHITECTURE_CONFIG: dict[str, Any] = {
    "d_model": 256,
    "encoder_layers": 4,
    "predictor_layers": 2,
    "nhead": 8,
    "dim_feedforward": 1024,
    "dropout": 0.1,
    "use_cls": True,
    "elo_scale": 3000.0,
}


def resolve_architecture_config(user: dict[str, Any] | None) -> dict[str, Any]:
    cfg = {**DEFAULT_ARCHITECTURE_CONFIG, **(user or {})}
    if cfg.pop("num_negatives_k", None) is not None:
        warnings.warn(
            "architecture.config num_negatives_k is ignored; use stages[*].hard_negatives "
            "n_hard + m_random as K.",
            UserWarning,
            stacklevel=2,
        )
    cfg["d_model"] = int(cfg["d_model"])
    cfg["encoder_layers"] = int(cfg["encoder_layers"])
    cfg["predictor_layers"] = int(cfg["predictor_layers"])
    cfg["nhead"] = int(cfg["nhead"])
    cfg["dim_feedforward"] = int(cfg["dim_feedforward"])
    cfg["dropout"] = float(cfg["dropout"])
    cfg["use_cls"] = bool(cfg["use_cls"])
    cfg["elo_scale"] = float(cfg["elo_scale"])
    if cfg["d_model"] % cfg["nhead"] != 0:
        raise ValueError("d_model must be divisible by nhead")
    return cfg


class BoardEncoder(nn.Module):
    """64 square tokens (+ optional CLS), TransformerEncoder, raw d_model latent."""

    def __init__(
        self,
        *,
        d_model: int,
        n_layers: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        use_cls: bool,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_cls = use_cls
        n_squares = BOARD_HEIGHT * BOARD_WIDTH
        self.square_embed = nn.Linear(BOARD_CHANNELS, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_squares, d_model))
        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        else:
            self.cls_token = None
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if use_cls:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        # board: (B, 8, 8, C) -> (B, 64, C)
        b = board.shape[0]
        x = board.reshape(b, BOARD_HEIGHT * BOARD_WIDTH, BOARD_CHANNELS).float()
        x = self.square_embed(x) + self.pos_embed
        if self.use_cls:
            cls = self.cls_token.expand(b, -1, -1)
            x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        if self.use_cls:
            out = x[:, 0, :]
        else:
            out = x.mean(dim=1)
        return out


class PredictorHead(nn.Module):
    """Elo-conditioned 2-layer Transformer on a single token; raw d_model output."""

    def __init__(
        self,
        *,
        d_model: int,
        n_layers: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        elo_scale: float,
    ) -> None:
        super().__init__()
        self.elo_scale = elo_scale
        self.elo_proj = nn.Linear(1, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, z_online: torch.Tensor, elo: torch.Tensor) -> torch.Tensor:
        # z_online: (B, D); elo: (B,)
        e = (elo.float().unsqueeze(-1) / self.elo_scale).clamp(-10.0, 10.0)
        fused = z_online + self.elo_proj(e)
        x = fused.unsqueeze(1)
        x = self.encoder(x)
        out = x.squeeze(1)
        return out


class ChessJEPA(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg["d_model"]
        self.encoder_online = BoardEncoder(
            d_model=d,
            n_layers=cfg["encoder_layers"],
            nhead=cfg["nhead"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
            use_cls=cfg["use_cls"],
        )
        self.encoder_target = copy.deepcopy(self.encoder_online)
        for p in self.encoder_target.parameters():
            p.requires_grad = False
        self.predictor = PredictorHead(
            d_model=d,
            n_layers=cfg["predictor_layers"],
            nhead=cfg["nhead"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
            elo_scale=cfg["elo_scale"],
        )

    def train(self, mode: bool = True) -> ChessJEPA:
        super().train(mode)
        self.encoder_target.eval()
        return self

    def trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.encoder_online.parameters()) + list(self.predictor.parameters())

    @torch.no_grad()
    def init_target_from_online(self) -> None:
        self.encoder_target.load_state_dict(self.encoder_online.state_dict())

    @torch.no_grad()
    def ema_update_target(self, momentum: float) -> None:
        m = float(momentum)
        for p_t, p_o in zip(self.encoder_target.parameters(), self.encoder_online.parameters(), strict=True):
            p_t.data.mul_(m).add_(p_o.data, alpha=1.0 - m)

    def forward_online(
        self,
        board_t: torch.Tensor,
        elo: torch.Tensor,
        from_sq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del from_sq  # v1 predictor is Elo-only; optional for shared benchmark API
        z_online = self.encoder_online(board_t)
        z_hat = self.predictor(z_online, elo)
        return z_online, z_hat

    def forward_target(self, board: torch.Tensor) -> torch.Tensor:
        return self.encoder_target(board)

    def forward_target_stack(self, boards_bk: torch.Tensor) -> torch.Tensor:
        """boards_bk: (B, K, 8, 8, C) -> (B, K, D)"""
        b, k, h, w, c = boards_bk.shape
        flat = boards_bk.reshape(b * k, h, w, c)
        z = self.forward_target(flat)
        return z.reshape(b, k, -1)


def jepa_triplet_vicreg_loss(
    z_online: torch.Tensor,
    z_hat: torch.Tensor,
    z_pos: torch.Tensor,
    z_negs: torch.Tensor,
    *,
    margin_alpha: float,
    vicreg_var_coef: float,
    vicreg_std_target: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Squared L2 geometry. z_negs: (B, K, D).
    d_pos = ||z_hat - z_pos||^2, d_neg_k = ||z_hat - z_neg_k||^2.
    Mining: active negatives satisfy d_neg < d_pos + margin.
    mean_n_neg_within_margin: batch mean count of negatives satisfying that inequality (per row, among K).
    Selection: easiest active (largest active d_neg); if none active, choose hardest inactive
    (smallest d_neg overall), yielding zero triplet term and only VICReg contribution.
    Loss: mean relu(d_pos - d_neg_sel + margin). VICReg variance on z_online unchanged.
    """
    # Under CUDA autocast, fp16 can break large sentinels; keep this block in full precision.
    amp_off = torch.amp.autocast("cuda", enabled=False) if z_hat.is_cuda else nullcontext()
    with amp_off:
        z_online = z_online.float()
        z_hat = z_hat.float()
        z_pos = z_pos.float()
        z_negs = z_negs.float()

        d_pos = (z_hat - z_pos).pow(2).sum(dim=-1)
        d_negs = (z_negs - z_hat.unsqueeze(1)).pow(2).sum(dim=-1)

        margin = float(margin_alpha)
        sp = d_pos.unsqueeze(-1)
        active = d_negs < (sp + margin)
        neg_inf = torch.full_like(d_negs, float("-inf"))
        masked_active = torch.where(active, d_negs, neg_inf)
        has_active = active.any(dim=-1)
        idx_easiest_active = masked_active.argmax(dim=-1)
        idx_hardest_inactive = d_negs.argmin(dim=-1)
        idx = torch.where(has_active, idx_easiest_active, idx_hardest_inactive)
        b_idx = torch.arange(z_hat.shape[0], device=z_hat.device, dtype=torch.long)
        d_neg_sel = d_negs[b_idx, idx]

        triplet_i = F.relu(d_pos - d_neg_sel + margin)
        triplet = triplet_i.mean()

        pct_active = has_active.float().mean() * 100.0
        mean_n_neg_within_margin = active.float().sum(dim=-1).mean()
        max_d_negs = d_negs.max(dim=1).values
        pct_pos_beats_hardest_neg = (d_pos < max_d_negs).float().mean() * 100.0

        if z_online.shape[0] < 2:
            vic = z_online.new_zeros(())
            vicreg_std_mean = z_online.new_zeros(())
        else:
            std = z_online.std(dim=0, unbiased=False)
            vic = F.relu(float(vicreg_std_target) - std).mean()
            vicreg_std_mean = std.mean()

        loss = triplet + float(vicreg_var_coef) * vic
        metrics = {
            "triplet": float(triplet.detach()),
            "vicreg": float(vic.detach()),
            "loss": float(loss.detach()),
            "pct_active": float(pct_active.detach()),
            "mean_n_neg_within_margin": float(mean_n_neg_within_margin.detach()),
            "pct_pos_beats_hardest_neg": float(pct_pos_beats_hardest_neg.detach()),
            "vicreg_std_mean": float(vicreg_std_mean.detach()),
        }
    return loss, metrics


class ChessJEPABuilder:
    @staticmethod
    def build(architecture_config: dict[str, Any] | None) -> ChessJEPA:
        cfg = resolve_architecture_config(architecture_config)
        m = ChessJEPA(cfg)
        m.init_target_from_online()
        return m
