"""
Chess-JEPA v3: encoder exposes CLS/GAP (B, d_model). JEPA MLP uses that plus
learned from/to slot embeddings (two tables of 64). From/to CE heads use global + indices.

Move-head prefix uses ``move_head_prefix_leak`` λ in ``[0, 1]`` (training config per stage):
``(1-λ)·detach(z[:,:P]) + λ·z[:,:P]`` with tail ``z[:,P:]`` unchanged. λ=0 matches full
prefix stopgrad for CE; λ=1 matches no prefix stopgrad.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn

from jepa2.config import BOARD_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH

ARCHITECTURE_ID = "chess_jepa_v3"

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
    cfg["move_head_prefix_leak"] = float(cfg.get("move_head_prefix_leak", 0.0))
    if not (0.0 <= cfg["move_head_prefix_leak"] <= 1.0):
        raise ValueError(f"move_head_prefix_leak must be in [0, 1], got {cfg['move_head_prefix_leak']}")
    if cfg["d_model"] % cfg["nhead"] != 0:
        raise ValueError("d_model must be divisible by nhead")
    if cfg["predictor_prefix_dims"] < 1 or cfg["predictor_prefix_dims"] > cfg["d_model"]:
        raise ValueError(
            f"predictor_prefix_dims must be in [1, d_model] (JEPA predictor output width / target prefix); "
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
    return cfg


def _mlp_gelu(
    in_dim: int,
    hidden: int,
    depth: int,
    out_dim: int,
) -> nn.Sequential:
    """``depth-1`` blocks of Linear+GELU then Linear to ``out_dim`` (same pattern as ``JepaMlpPredictor``)."""
    layers: list[nn.Module] = []
    d = int(in_dim)
    h = int(hidden)
    for _ in range(int(depth) - 1):
        layers.append(nn.Linear(d, h))
        layers.append(nn.GELU())
        d = h
    layers.append(nn.Linear(d, int(out_dim)))
    return nn.Sequential(*layers)


class BoardEncoderV3(nn.Module):
    """64 square tokens (+ optional CLS). ``forward`` returns global only."""

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

    def forward_with_tokens(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = board.shape[0]
        x = board.reshape(b, BOARD_HEIGHT * BOARD_WIDTH, BOARD_CHANNELS).float()
        x = self.square_embed(x) + self.pos_embed
        if self.use_cls:
            assert self.cls_token is not None
            cls = self.cls_token.expand(b, -1, -1)
            x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        if self.use_cls:
            z_global = x[:, 0, :]
            tokens = x[:, 1:, :]
        else:
            tokens = x
            z_global = tokens.mean(dim=1)
        return z_global, tokens

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        z_global, _ = self.forward_with_tokens(board)
        return z_global


class JepaMlpPredictor(nn.Module):
    """MLP on concat(z_global, e_from, e_to) -> P dimensions (invariance target prefix)."""

    def __init__(
        self,
        *,
        output_dims: int,
        d_model: int,
        square_embed_dim: int,
        hidden: int,
        depth: int,
    ) -> None:
        super().__init__()
        self.output_dims = int(output_dims)
        in_dim = int(d_model) + 2 * int(square_embed_dim)
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(int(depth) - 1):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.GELU())
            d = hidden
        layers.append(nn.Linear(d, self.output_dims))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        z_global: torch.Tensor,
        e_from: torch.Tensor,
        e_to: torch.Tensor,
    ) -> torch.Tensor:
        """z_global (B, D), e_from / e_to (B, E)."""
        return self.net(torch.cat([z_global, e_from, e_to], dim=-1))


class FromSquareHead(nn.Module):
    """64-way from-square logits: MLP on global representation only."""

    def __init__(self, d_model: int, hidden: int, depth: int) -> None:
        super().__init__()
        self.net = _mlp_gelu(d_model, hidden, depth, 64)

    def forward(self, z_global: torch.Tensor) -> torch.Tensor:
        """z_global (B, D) -> logits (B, 64)."""
        return self.net(z_global)


class ToSquareHead(nn.Module):
    """64-way to-square logits: from index embedding + MLP on concat(z_global, e_from)."""

    def __init__(self, d_model: int, hidden: int, depth: int) -> None:
        super().__init__()
        self.from_embed = nn.Embedding(64, d_model)
        self.net = _mlp_gelu(2 * d_model, hidden, depth, 64)

    def forward(self, z_global: torch.Tensor, from_sq: torch.Tensor) -> torch.Tensor:
        """z_global (B, D), from_sq (B,) int64 -> logits (B, 64)."""
        e = self.from_embed(from_sq.long())
        x = torch.cat([z_global, e], dim=-1)
        return self.net(x)


class ChessJEPAV3(nn.Module):
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
        for p in self.encoder_target.parameters():
            p.requires_grad = False
        self.jepa_from_embed = nn.Embedding(64, e)
        self.jepa_to_embed = nn.Embedding(64, e)
        self.from_slot_unknown = nn.Parameter(torch.zeros(e))
        self.jepa_predictor = JepaMlpPredictor(
            output_dims=int(cfg["predictor_prefix_dims"]),
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

    def train(self, mode: bool = True) -> ChessJEPAV3:
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
        """Public online encoding: CLS or GAP only, shape (B, d_model)."""
        return self.encoder_online(board)

    def encode_online_with_jepa(
        self,
        board: torch.Tensor,
        from_sq: torch.Tensor,
        to_sq: torch.Tensor,
        *,
        from_sq_unk: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Same single ``encoder_online(board)`` as other heads; returns ``(z_global, z_hat)``.
        """
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


class ChessJEPAV3Builder:
    @staticmethod
    def build(architecture_config: dict[str, Any] | None) -> ChessJEPAV3:
        cfg = resolve_architecture_config(architecture_config)
        m = ChessJEPAV3(cfg)
        m.init_target_from_online()
        return m
