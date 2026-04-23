"""
Chess-JEPA v4: same encoder / JEPA / from-to heads as v3, plus optional auxiliary
tasks (board reconstruction, meta rights) on the full global embedding.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa3.architectures.board_probe_labels import meta_targets_from_board, piece_labels_64_from_board
from jepa3.architectures.chess_jepa_v3 import (
    BoardEncoderV3,
    FromSquareHead,
    JepaMlpPredictor,
    ToSquareHead,
    _mlp_gelu,
)

ARCHITECTURE_ID = "chess_jepa_v4"

DEFAULT_ARCHITECTURE_CONFIG: dict[str, Any] = {
    "d_model": 256,
    "encoder_layers": 4,
    "nhead": 8,
    "dim_feedforward": 1024,
    "dropout": 0.1,
    "use_cls": True,
    "jepa_square_embed_dim": 64,
    "predictor_hidden": 512,
    "predictor_depth": 2,
    "from_to_head_hidden": 256,
    "from_to_head_depth": 2,
    "aux_board_recon_hidden": 256,
    "aux_meta_hidden": 128,
}


def resolve_architecture_config(user: dict[str, Any] | None) -> dict[str, Any]:
    cfg = {**DEFAULT_ARCHITECTURE_CONFIG, **(user or {})}
    cfg["d_model"] = int(cfg["d_model"])
    cfg["encoder_layers"] = int(cfg["encoder_layers"])
    cfg["nhead"] = int(cfg["nhead"])
    cfg["dim_feedforward"] = int(cfg["dim_feedforward"])
    cfg["dropout"] = float(cfg["dropout"])
    cfg["use_cls"] = bool(cfg["use_cls"])
    cfg["jepa_square_embed_dim"] = int(cfg["jepa_square_embed_dim"])
    cfg["predictor_hidden"] = int(cfg["predictor_hidden"])
    cfg["predictor_depth"] = int(cfg["predictor_depth"])
    cfg["from_to_head_hidden"] = int(cfg["from_to_head_hidden"])
    cfg["from_to_head_depth"] = int(cfg["from_to_head_depth"])
    cfg["aux_board_recon_hidden"] = int(cfg["aux_board_recon_hidden"])
    cfg["aux_meta_hidden"] = int(cfg["aux_meta_hidden"])
    if cfg["d_model"] % cfg["nhead"] != 0:
        raise ValueError("d_model must be divisible by nhead")
    if cfg["jepa_square_embed_dim"] < 1:
        raise ValueError(f"jepa_square_embed_dim must be >= 1 (got {cfg['jepa_square_embed_dim']})")
    if cfg["predictor_depth"] < 1:
        raise ValueError("predictor_depth must be >= 1")
    if cfg["from_to_head_hidden"] < 1:
        raise ValueError(f"from_to_head_hidden must be >= 1 (got {cfg['from_to_head_hidden']})")
    if cfg["from_to_head_depth"] < 1:
        raise ValueError(f"from_to_head_depth must be >= 1 (got {cfg['from_to_head_depth']})")
    if cfg["aux_board_recon_hidden"] < 1:
        raise ValueError("aux_board_recon_hidden must be >= 1")
    if cfg["aux_meta_hidden"] < 1:
        raise ValueError("aux_meta_hidden must be >= 1")
    return cfg


class BoardReconAux(nn.Module):
    """MLP on full global embedding -> (B, 64, 13) piece logits."""

    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.net = _mlp_gelu(d_model, hidden, 2, 64 * 13)

    def forward(self, z_global: torch.Tensor) -> torch.Tensor:
        logits_flat = self.net(z_global)
        return logits_flat.view(z_global.shape[0], 64, 13)


class MetaAux(nn.Module):
    """Turn (BCE), castling x4 (BCE), en passant 65-class CE."""

    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.backbone = _mlp_gelu(d_model, hidden, 2, hidden)
        self.head_turn = nn.Linear(hidden, 1)
        self.head_castle = nn.Linear(hidden, 4)
        self.head_ep = nn.Linear(hidden, 65)

    def forward(self, z_global: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(z_global)
        return self.head_turn(h), self.head_castle(h), self.head_ep(h)


class ChessJEPAV4(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg["d_model"]
        e = int(cfg["jepa_square_embed_dim"])
        enc_kw = dict(
            d_model=d,
            n_layers=cfg["encoder_layers"],
            nhead=cfg["nhead"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
            use_cls=cfg["use_cls"],
        )
        self.encoder_online = BoardEncoderV3(**enc_kw)
        self.encoder_target = copy.deepcopy(self.encoder_online)
        for par in self.encoder_target.parameters():
            par.requires_grad = False
        self.jepa_from_embed = nn.Embedding(64, e)
        self.jepa_to_embed = nn.Embedding(64, e)
        self.from_slot_unknown = nn.Parameter(torch.zeros(e))
        self.jepa_predictor = JepaMlpPredictor(
            output_dims=d,
            d_model=d,
            square_embed_dim=e,
            hidden=int(cfg["predictor_hidden"]),
            depth=int(cfg["predictor_depth"]),
        )
        self.from_square_head = FromSquareHead(
            d,
            hidden=int(cfg["from_to_head_hidden"]),
            depth=int(cfg["from_to_head_depth"]),
        )
        self.to_square_head = ToSquareHead(
            d,
            hidden=int(cfg["from_to_head_hidden"]),
            depth=int(cfg["from_to_head_depth"]),
        )
        self.aux_board_recon = BoardReconAux(d, int(cfg["aux_board_recon_hidden"]))
        self.aux_meta = MetaAux(d, int(cfg["aux_meta_hidden"]))

    def train(self, mode: bool = True) -> ChessJEPAV4:
        super().train(mode)
        self.encoder_target.eval()
        return self

    def trainable_parameters(self) -> list[nn.Parameter]:
        return (
            list(self.encoder_online.parameters())
            + list(self.jepa_predictor.parameters())
            + list(self.from_square_head.parameters())
            + list(self.to_square_head.parameters())
            + list(self.jepa_from_embed.parameters())
            + list(self.jepa_to_embed.parameters())
            + [self.from_slot_unknown]
            + list(self.aux_board_recon.parameters())
            + list(self.aux_meta.parameters())
        )

    @torch.no_grad()
    def init_target_from_online(self) -> None:
        self.encoder_target.load_state_dict(self.encoder_online.state_dict())

    @torch.no_grad()
    def ema_update_target(self, momentum: float) -> None:
        m = float(momentum)
        for p_t, p_o in zip(self.encoder_target.parameters(), self.encoder_online.parameters(), strict=True):
            p_t.data.mul_(m).add_(p_o.data, alpha=1.0 - m)

    def encode_online(self, board: torch.Tensor) -> torch.Tensor:
        return self.encoder_online(board)

    def encode_online_with_jepa(
        self,
        board: torch.Tensor,
        from_sq: torch.Tensor,
        to_sq: torch.Tensor,
        *,
        from_sq_unk: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder_online(board)
        e_from = self.jepa_from_embed(from_sq.long())
        if from_sq_unk is not None and bool(from_sq_unk.any()):
            unk = self.from_slot_unknown.to(dtype=e_from.dtype, device=e_from.device).view(1, -1).expand_as(e_from)
            e_from = torch.where(from_sq_unk.unsqueeze(-1), unk, e_from)
        e_to = self.jepa_to_embed(to_sq.long())
        z_hat = self.jepa_predictor(z, e_from, e_to)
        return z, z_hat

    def encode_target_global(self, board: torch.Tensor) -> torch.Tensor:
        return self.encoder_target(board)

    def forward_from_logits(self, z_global: torch.Tensor) -> torch.Tensor:
        return self.from_square_head(z_global)

    def forward_to_logits(self, z_global: torch.Tensor, from_sq: torch.Tensor) -> torch.Tensor:
        return self.to_square_head(z_global, from_sq)

    def forward_prefix_aux_losses(
        self,
        board: torch.Tensor,
        z_global: torch.Tensor,
        *,
        compute_board_recon: bool,
        compute_meta: bool,
    ) -> dict[str, torch.Tensor]:
        """Auxiliary losses on the full global embedding; gradients flow into ``encoder_online``.

        Returns ``aux_board_recon`` (scalar CE) and ``aux_board_recon_top1`` (mean % correct
        squares, 0--100) when board recon runs. When meta runs: ``aux_meta`` plus
        ``aux_meta_turn_top1``, ``aux_meta_castle_top1``, ``aux_meta_ep_top1``, and
        ``aux_meta_top1`` (mean of the three, 0--100 each).
        """
        out: dict[str, torch.Tensor] = {}
        if compute_board_recon:
            logits = self.aux_board_recon(z_global)
            labels = piece_labels_64_from_board(board)
            ce = F.cross_entropy(logits.reshape(-1, 13), labels.reshape(-1))
            out["aux_board_recon"] = ce
            pred_sq = logits.argmax(dim=-1)
            br_top1 = (pred_sq == labels).float().mean() * 100.0
            out["aux_board_recon_top1"] = br_top1
        if compute_meta:
            turn_l, castle_l, ep_l = self.aux_meta(z_global)
            meta = meta_targets_from_board(board)
            loss_turn = F.binary_cross_entropy_with_logits(turn_l.squeeze(-1), meta["turn"])
            loss_castle = F.binary_cross_entropy_with_logits(castle_l, meta["castle"])
            loss_ep = F.cross_entropy(ep_l, meta["ep_class"])
            out["aux_meta"] = loss_turn + loss_castle + loss_ep
            turn_pred = (torch.sigmoid(turn_l.squeeze(-1)) > 0.5).float()
            turn_top1 = (turn_pred == meta["turn"]).float().mean() * 100.0
            castle_pred = (torch.sigmoid(castle_l) > 0.5).float()
            castle_top1 = (castle_pred == meta["castle"]).all(dim=-1).float().mean() * 100.0
            ep_top1 = (ep_l.argmax(dim=-1) == meta["ep_class"]).float().mean() * 100.0
            out["aux_meta_turn_top1"] = turn_top1
            out["aux_meta_castle_top1"] = castle_top1
            out["aux_meta_ep_top1"] = ep_top1
            out["aux_meta_top1"] = (turn_top1 + castle_top1 + ep_top1) / 3.0
        return out


class ChessJEPAV4Builder:
    @staticmethod
    def build(architecture_config: dict[str, Any] | None) -> ChessJEPAV4:
        cfg = resolve_architecture_config(architecture_config)
        m = ChessJEPAV4(cfg)
        m.init_target_from_online()
        return m
