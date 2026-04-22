"""
Chess-JEPA v4: same encoder / JEPA / from-to heads as v3, but move-head prefix uses a
scalar ``move_head_prefix_leak`` blend instead of a binary stopgrad, plus optional
detached-prefix auxiliary probes (board reconstruction, meta rights).
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
    "predictor_prefix_dims": 64,
    "jepa_square_embed_dim": 64,
    "predictor_hidden": 512,
    "predictor_depth": 2,
    "from_to_head_hidden": 256,
    "from_to_head_depth": 2,
    "move_head_prefix_leak": 0.0,
    "probe_board_recon_hidden": 256,
    "probe_meta_hidden": 128,
}


def resolve_architecture_config(user: dict[str, Any] | None) -> dict[str, Any]:
    cfg = {**DEFAULT_ARCHITECTURE_CONFIG, **(user or {})}
    cfg["d_model"] = int(cfg["d_model"])
    cfg["encoder_layers"] = int(cfg["encoder_layers"])
    cfg["nhead"] = int(cfg["nhead"])
    cfg["dim_feedforward"] = int(cfg["dim_feedforward"])
    cfg["dropout"] = float(cfg["dropout"])
    cfg["use_cls"] = bool(cfg["use_cls"])
    cfg["predictor_prefix_dims"] = int(cfg["predictor_prefix_dims"])
    cfg["jepa_square_embed_dim"] = int(cfg["jepa_square_embed_dim"])
    cfg["predictor_hidden"] = int(cfg["predictor_hidden"])
    cfg["predictor_depth"] = int(cfg["predictor_depth"])
    cfg["from_to_head_hidden"] = int(cfg["from_to_head_hidden"])
    cfg["from_to_head_depth"] = int(cfg["from_to_head_depth"])
    cfg["move_head_prefix_leak"] = float(cfg["move_head_prefix_leak"])
    cfg["probe_board_recon_hidden"] = int(cfg["probe_board_recon_hidden"])
    cfg["probe_meta_hidden"] = int(cfg["probe_meta_hidden"])
    if cfg["d_model"] % cfg["nhead"] != 0:
        raise ValueError("d_model must be divisible by nhead")
    if cfg["predictor_prefix_dims"] < 1 or cfg["predictor_prefix_dims"] > cfg["d_model"]:
        raise ValueError(
            f"predictor_prefix_dims must be in [1, d_model]; "
            f"got {cfg['predictor_prefix_dims']} vs d_model={cfg['d_model']}"
        )
    if cfg["jepa_square_embed_dim"] < 1:
        raise ValueError(f"jepa_square_embed_dim must be >= 1 (got {cfg['jepa_square_embed_dim']})")
    if cfg["predictor_depth"] < 1:
        raise ValueError("predictor_depth must be >= 1")
    if cfg["from_to_head_hidden"] < 1:
        raise ValueError(f"from_to_head_hidden must be >= 1 (got {cfg['from_to_head_hidden']})")
    if cfg["from_to_head_depth"] < 1:
        raise ValueError(f"from_to_head_depth must be >= 1 (got {cfg['from_to_head_depth']})")
    if not (0.0 <= cfg["move_head_prefix_leak"] <= 1.0):
        raise ValueError(f"move_head_prefix_leak must be in [0, 1], got {cfg['move_head_prefix_leak']}")
    if cfg["probe_board_recon_hidden"] < 1:
        raise ValueError("probe_board_recon_hidden must be >= 1")
    if cfg["probe_meta_hidden"] < 1:
        raise ValueError("probe_meta_hidden must be >= 1")
    return cfg


class BoardReconProbe(nn.Module):
    """Detached-prefix MLP -> (B, 64, 13) piece logits."""

    def __init__(self, prefix_dim: int, hidden: int) -> None:
        super().__init__()
        if prefix_dim != 64:
            raise ValueError(f"v4 board recon probe requires predictor_prefix_dims==64, got {prefix_dim}")
        self.net = _mlp_gelu(prefix_dim, hidden, 2, 64 * 13)

    def forward(self, z_prefix_det: torch.Tensor) -> torch.Tensor:
        logits_flat = self.net(z_prefix_det)
        return logits_flat.view(z_prefix_det.shape[0], 64, 13)


class MetaProbe(nn.Module):
    """Turn (BCE), castling x4 (BCE), en passant 65-class CE."""

    def __init__(self, prefix_dim: int, hidden: int) -> None:
        super().__init__()
        self.backbone = _mlp_gelu(prefix_dim, hidden, 2, hidden)
        self.head_turn = nn.Linear(hidden, 1)
        self.head_castle = nn.Linear(hidden, 4)
        self.head_ep = nn.Linear(hidden, 65)

    def forward(self, z_prefix_det: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(z_prefix_det)
        return self.head_turn(h), self.head_castle(h), self.head_ep(h)


class ChessJEPAV4(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg["d_model"]
        e = int(cfg["jepa_square_embed_dim"])
        p = int(cfg["predictor_prefix_dims"])
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
            output_dims=p,
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
        self.probe_board_recon = BoardReconProbe(p, int(cfg["probe_board_recon_hidden"]))
        self.probe_meta = MetaProbe(p, int(cfg["probe_meta_hidden"]))

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
            + list(self.probe_board_recon.parameters())
            + list(self.probe_meta.parameters())
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

    def _z_global_for_move_heads(self, z_global: torch.Tensor, *, leak: float) -> torch.Tensor:
        lam = float(leak)
        lam = max(0.0, min(1.0, lam))
        p = int(self.cfg["predictor_prefix_dims"])
        d = int(z_global.shape[-1])
        if d <= p:
            return z_global
        pref = z_global[:, :p]
        blended = (1.0 - lam) * pref.detach() + lam * pref
        return torch.cat([blended, z_global[:, p:]], dim=-1)

    def forward_from_logits(self, z_global: torch.Tensor, *, move_head_prefix_leak: float | None = None) -> torch.Tensor:
        lam = float(self.cfg["move_head_prefix_leak"]) if move_head_prefix_leak is None else float(move_head_prefix_leak)
        z_in = self._z_global_for_move_heads(z_global, leak=lam)
        return self.from_square_head(z_in)

    def forward_to_logits(
        self, z_global: torch.Tensor, from_sq: torch.Tensor, *, move_head_prefix_leak: float | None = None
    ) -> torch.Tensor:
        lam = float(self.cfg["move_head_prefix_leak"]) if move_head_prefix_leak is None else float(move_head_prefix_leak)
        z_in = self._z_global_for_move_heads(z_global, leak=lam)
        return self.to_square_head(z_in, from_sq)

    def forward_aux_losses(
        self,
        board: torch.Tensor,
        z_global: torch.Tensor,
        *,
        compute_board_recon: bool,
        compute_meta: bool,
    ) -> dict[str, torch.Tensor]:
        """Detached-prefix probes only; encoder does not receive grads from these losses."""
        out: dict[str, torch.Tensor] = {}
        p = int(self.cfg["predictor_prefix_dims"])
        if p != 64:
            return out
        z_prefix = z_global[:, :p].detach()
        if compute_board_recon:
            logits = self.probe_board_recon(z_prefix)
            labels = piece_labels_64_from_board(board)
            ce = F.cross_entropy(logits.reshape(-1, 13), labels.reshape(-1))
            out["probe_board_recon"] = ce
        if compute_meta:
            turn_l, castle_l, ep_l = self.probe_meta(z_prefix)
            meta = meta_targets_from_board(board)
            loss_turn = F.binary_cross_entropy_with_logits(turn_l.squeeze(-1), meta["turn"])
            loss_castle = F.binary_cross_entropy_with_logits(castle_l, meta["castle"])
            loss_ep = F.cross_entropy(ep_l, meta["ep_class"])
            out["probe_meta"] = loss_turn + loss_castle + loss_ep
        return out


class ChessJEPAV4Builder:
    @staticmethod
    def build(architecture_config: dict[str, Any] | None) -> ChessJEPAV4:
        cfg = resolve_architecture_config(architecture_config)
        m = ChessJEPAV4(cfg)
        m.init_target_from_online()
        return m
