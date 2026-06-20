"""
Chess world-model v1: jepa3 BoardEncoderV3 (CLS + 64 patches), Transformer patch JEPA predictor
(four action tokens + patches), patch-pointer from-square CE head, optional reconstruction heads.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn

from jepa3.architectures.chess_jepa_v3 import BoardEncoderV3, _mlp_gelu
from jepa3.board_square_categories import NUM_SQUARE_CATEGORIES

from world_model.action_tokens import moved_placed_categories_from_move
from world_model.architectures.jepa_patch_transformer_predictor import JepaPatchTransformerPredictor
from world_model.architectures.patch_pointer_from_head import PatchPointerFromHead

ARCHITECTURE_ID = "chess_world_model_v1"

DEFAULT_ARCHITECTURE_CONFIG: dict[str, Any] = {
    "d_model": 256,
    "encoder_layers": 4,
    "nhead": 8,
    "dim_feedforward": 1024,
    "dropout": 0.1,
    "jepa_square_embed_dim": 64,
    "predictor_encoder_layers": 2,
    "predictor_nhead": 8,
    "predictor_dim_feedforward": 1024,
    "predictor_dropout": 0.1,
    "from_to_head_hidden": 256,
    "from_to_head_depth": 2,
    "from_pointer_dim": 256,
    "from_pointer_elo_dim": 32,
    "recon_piece_head_hidden": 256,
    "recon_turn_head_hidden": 256,
    "recon_can_move_head_hidden": 256,
}


def resolve_architecture_config(user: dict[str, Any] | None) -> dict[str, Any]:
    cfg = {**DEFAULT_ARCHITECTURE_CONFIG, **(user or {})}
    cfg["d_model"] = int(cfg["d_model"])
    cfg["encoder_layers"] = int(cfg["encoder_layers"])
    cfg["nhead"] = int(cfg["nhead"])
    cfg["dim_feedforward"] = int(cfg["dim_feedforward"])
    cfg["dropout"] = float(cfg["dropout"])
    cfg["jepa_square_embed_dim"] = int(cfg["jepa_square_embed_dim"])
    cfg["predictor_encoder_layers"] = int(cfg["predictor_encoder_layers"])
    cfg["predictor_nhead"] = int(cfg["predictor_nhead"])
    cfg["predictor_dim_feedforward"] = int(cfg["predictor_dim_feedforward"])
    cfg["predictor_dropout"] = float(cfg["predictor_dropout"])
    cfg["from_to_head_hidden"] = int(cfg["from_to_head_hidden"])
    cfg["from_to_head_depth"] = int(cfg["from_to_head_depth"])
    if cfg["d_model"] % cfg["nhead"] != 0:
        raise ValueError("d_model must be divisible by nhead")
    if cfg["d_model"] % cfg["predictor_nhead"] != 0:
        raise ValueError("d_model must be divisible by predictor_nhead")
    if cfg["jepa_square_embed_dim"] < 1:
        raise ValueError(f"jepa_square_embed_dim must be >= 1 (got {cfg['jepa_square_embed_dim']})")
    if cfg["predictor_encoder_layers"] < 1:
        raise ValueError(f"predictor_encoder_layers must be >= 1 (got {cfg['predictor_encoder_layers']})")
    if cfg["predictor_dim_feedforward"] < 1:
        raise ValueError(f"predictor_dim_feedforward must be >= 1 (got {cfg['predictor_dim_feedforward']})")
    if cfg["from_to_head_hidden"] < 1:
        raise ValueError(f"from_to_head_hidden must be >= 1 (got {cfg['from_to_head_hidden']})")
    if cfg["from_to_head_depth"] < 1:
        raise ValueError(f"from_to_head_depth must be >= 1 (got {cfg['from_to_head_depth']})")
    cfg["from_pointer_dim"] = int(cfg["from_pointer_dim"])
    cfg["from_pointer_elo_dim"] = int(cfg["from_pointer_elo_dim"])
    if cfg["from_pointer_dim"] < 1:
        raise ValueError(f"from_pointer_dim must be >= 1 (got {cfg['from_pointer_dim']})")
    if cfg["from_pointer_elo_dim"] < 0:
        raise ValueError(f"from_pointer_elo_dim must be >= 0 (got {cfg['from_pointer_elo_dim']})")
    cfg["recon_piece_head_hidden"] = int(cfg["recon_piece_head_hidden"])
    cfg["recon_turn_head_hidden"] = int(cfg["recon_turn_head_hidden"])
    if cfg["recon_piece_head_hidden"] < 1:
        raise ValueError(f"recon_piece_head_hidden must be >= 1 (got {cfg['recon_piece_head_hidden']})")
    if cfg["recon_turn_head_hidden"] < 1:
        raise ValueError(f"recon_turn_head_hidden must be >= 1 (got {cfg['recon_turn_head_hidden']})")
    cfg["recon_can_move_head_hidden"] = int(cfg["recon_can_move_head_hidden"])
    if cfg["recon_can_move_head_hidden"] < 1:
        raise ValueError(f"recon_can_move_head_hidden must be >= 1 (got {cfg['recon_can_move_head_hidden']})")
    return cfg


class ChessWorldModelV1(nn.Module):
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
        )
        self.encoder_online = BoardEncoderV3(**enc_kw)
        self.encoder_target = copy.deepcopy(self.encoder_online)
        for par in self.encoder_target.parameters():
            par.requires_grad = False
        self.jepa_from_embed = nn.Embedding(64, e)
        self.jepa_to_embed = nn.Embedding(64, e)
        self.from_slot_unknown = nn.Parameter(torch.zeros(e))
        self.patch_predictor = JepaPatchTransformerPredictor(
            d_model=d,
            square_embed_dim=e,
            n_layers=int(cfg["predictor_encoder_layers"]),
            nhead=int(cfg["predictor_nhead"]),
            dim_feedforward=int(cfg["predictor_dim_feedforward"]),
            dropout=float(cfg["predictor_dropout"]),
        )
        # Query MLP width/depth reuse from_to_head_* (historical name).
        self.from_square_head = PatchPointerFromHead(
            d_model=d,
            pointer_dim=int(cfg["from_pointer_dim"]),
            query_hidden=int(cfg["from_to_head_hidden"]),
            query_depth=int(cfg["from_to_head_depth"]),
            elo_dim=int(cfg["from_pointer_elo_dim"]),
        )
        self.piece_recon_head = _mlp_gelu(
            d,
            int(cfg["recon_piece_head_hidden"]),
            2,
            NUM_SQUARE_CATEGORIES,
        )
        self.turn_recon_head = _mlp_gelu(
            d,
            int(cfg["recon_turn_head_hidden"]),
            2,
            2,
        )
        self.can_move_recon_head = _mlp_gelu(
            d,
            int(cfg["recon_can_move_head_hidden"]),
            2,
            2,
        )

    def train(self, mode: bool = True) -> ChessWorldModelV1:
        super().train(mode)
        self.encoder_target.eval()
        return self

    def trainable_parameters(self) -> list[nn.Parameter]:
        return (
            list(self.encoder_online.parameters())
            + list(self.patch_predictor.parameters())
            + list(self.from_square_head.parameters())
            + list(self.jepa_from_embed.parameters())
            + list(self.jepa_to_embed.parameters())
            + list(self.piece_recon_head.parameters())
            + list(self.turn_recon_head.parameters())
            + list(self.can_move_recon_head.parameters())
            + [self.from_slot_unknown]
        )

    @torch.no_grad()
    def init_target_from_online(self) -> None:
        self.encoder_target.load_state_dict(self.encoder_online.state_dict())

    @torch.no_grad()
    def ema_update_target(self, momentum: float) -> None:
        m = float(momentum)
        for p_t, p_o in zip(self.encoder_target.parameters(), self.encoder_online.parameters(), strict=True):
            p_t.data.mul_(m).add_(p_o.data, alpha=1.0 - m)

    def encode_online_with_jepa_and_patches(
        self,
        board_pre: torch.Tensor,
        board_post: torch.Tensor,
        from_sq: torch.Tensor,
        to_sq: torch.Tensor,
        *,
        from_sq_unk: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(z_global, patch_tokens_online, patch_hat)`` — patch-only JEPA predictor."""
        z_glob, patch_on = self.encoder_online.forward_with_tokens(board_pre)
        e_from = self.jepa_from_embed(from_sq.long())
        if from_sq_unk is not None and bool(from_sq_unk.any()):
            unk = self.from_slot_unknown.to(dtype=e_from.dtype, device=e_from.device).view(1, -1).expand_as(e_from)
            e_from = torch.where(from_sq_unk.unsqueeze(-1), unk, e_from)
        e_to = self.jepa_to_embed(to_sq.long())
        moved_cat, placed_cat = moved_placed_categories_from_move(board_pre, board_post, from_sq, to_sq)
        patch_hat = self.patch_predictor(patch_on, e_from, e_to, moved_cat, placed_cat)
        return z_glob, patch_on, patch_hat

    def encode_target_with_tokens(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder_target.forward_with_tokens(board)

    def forward_from_logits(
        self,
        z_global: torch.Tensor,
        patch_tokens: torch.Tensor,
        elo: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.from_square_head(z_global, patch_tokens, elo)

    def forward_reconstruction_logits(
        self,
        z_global: torch.Tensor,
        patch_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Piece-category logits per patch; turn logits from CLS; can-move logits per patch."""
        if patch_tokens.ndim != 3:
            raise ValueError(f"patch_tokens must be (B, 64, d_model), got {tuple(patch_tokens.shape)}")
        b, n_sq, d_model = patch_tokens.shape
        if n_sq != 64 or d_model != self.cfg["d_model"]:
            raise ValueError(
                f"patch_tokens must be (B, 64, d_model={self.cfg['d_model']}), got {tuple(patch_tokens.shape)}"
            )
        flat = patch_tokens.reshape(b * n_sq, d_model)
        piece_logits = self.piece_recon_head(flat).view(b, n_sq, NUM_SQUARE_CATEGORIES)
        can_move_logits = self.can_move_recon_head(flat).view(b, n_sq, 2)
        turn_logits = self.turn_recon_head(z_global)
        return {
            "piece_logits": piece_logits,
            "turn_logits": turn_logits,
            "can_move_logits": can_move_logits,
        }


class ChessWorldModelV1Builder:
    @staticmethod
    def build(architecture_config: dict[str, Any] | None) -> ChessWorldModelV1:
        cfg = resolve_architecture_config(architecture_config)
        m = ChessWorldModelV1(cfg)
        m.init_target_from_online()
        return m
