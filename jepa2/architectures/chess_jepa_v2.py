"""
Chess-JEPA v2 (jepa2): same backbone as v1 — spatial-token encoder, EMA target twin, Elo-conditioned predictor.
Loss lives in ``jepa2.loss`` (CE + MSE + VICReg), not triplet.
"""

from __future__ import annotations

import copy
import warnings
from typing import Any

import torch
import torch.nn as nn

from jepa2.config import BOARD_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH

ARCHITECTURE_ID = "chess_jepa_v2"

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
            "architecture.config num_negatives_k is ignored for jepa2; use M_train / M_eval in spec defaults.",
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

    def forward_online(self, board_t: torch.Tensor, elo: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


class ChessJEPABuilder:
    @staticmethod
    def build(architecture_config: dict[str, Any] | None) -> ChessJEPA:
        cfg = resolve_architecture_config(architecture_config)
        m = ChessJEPA(cfg)
        m.init_target_from_online()
        return m
