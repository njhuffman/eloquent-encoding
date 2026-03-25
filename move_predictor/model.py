"""Dual GRU (white / black history) + turn embedding + MLP over candidate moves."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class MovePredictor(nn.Module):
    """
    hist_white / hist_black: (B, N, D) float, chronological within each color, right-padded;
    hist_white_len / hist_black_len: (B,) long — valid prefix lengths;
    side_to_move: (B,) long — 0 = white to move, 1 = black to move.

    GRU outputs are concatenated with the side to move first (current player), then opponent.
    forward(): K=3, returns logits (B, 3) for cross-entropy vs label in {0,1,2}.
    score_moves(): arbitrary K (e.g. all legal moves for mining).
    """

    def __init__(
        self,
        embedding_dim: int,
        history_n: int,
        move_emb_dim: int = 8,
        turn_emb_dim: int = 4,
        gru_hidden: int = 1,
        gru_num_layers: int = 1,
        mlp_hidden: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.history_n = history_n
        self.turn_emb_dim = turn_emb_dim
        self.gru_hidden = gru_hidden
        self.emb_from = nn.Embedding(64, move_emb_dim)
        self.emb_to = nn.Embedding(64, move_emb_dim)
        self.turn_emb = nn.Embedding(2, turn_emb_dim)
        self.gru_white = nn.GRU(
            embedding_dim,
            gru_hidden,
            num_layers=gru_num_layers,
            batch_first=True,
        )
        self.gru_black = nn.GRU(
            embedding_dim,
            gru_hidden,
            num_layers=gru_num_layers,
            batch_first=True,
        )
        mlp_in = embedding_dim + 2 * move_emb_dim + 2 * gru_hidden + turn_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def _style_from_hist(
        self,
        gru: nn.GRU,
        hist_emb: torch.Tensor,
        hist_len: torch.Tensor,
    ) -> torch.Tensor:
        b, n, d = hist_emb.shape
        lengths = hist_len.clamp(min=0).long()
        hidden = gru.hidden_size
        style = torch.zeros(b, hidden, device=hist_emb.device, dtype=hist_emb.dtype)
        nonzero = lengths > 0
        if nonzero.any():
            idx = nonzero.nonzero(as_tuple=False).squeeze(-1)
            sub_hist = hist_emb[idx]
            sub_len = lengths[idx]
            sorted_len, sort_idx = sub_len.sort(0, descending=True)
            sorted_hist = sub_hist[sort_idx]
            packed = pack_padded_sequence(
                sorted_hist,
                sorted_len.cpu(),
                batch_first=True,
                enforce_sorted=True,
            )
            _, h_n = gru(packed)
            h_sorted = h_n[-1]
            unsort = sort_idx.argsort(0)
            h_sub = h_sorted[unsort]
            style[idx] = h_sub
        return style

    def _ordered_style(
        self,
        hist_w: torch.Tensor,
        hist_b: torch.Tensor,
        hlen_w: torch.Tensor,
        hlen_b: torch.Tensor,
        side_to_move: torch.Tensor,
    ) -> torch.Tensor:
        style_w = self._style_from_hist(self.gru_white, hist_w, hlen_w)
        style_b = self._style_from_hist(self.gru_black, hist_b, hlen_b)
        is_white = (side_to_move == 0).unsqueeze(-1)
        first = torch.where(is_white, style_w, style_b)
        second = torch.where(is_white, style_b, style_w)
        return torch.cat([first, second], dim=-1)

    def score_moves(
        self,
        cur_emb: torch.Tensor,
        hist_w: torch.Tensor,
        hist_b: torch.Tensor,
        hlen_w: torch.Tensor,
        hlen_b: torch.Tensor,
        side_to_move: torch.Tensor,
        from_sq: torch.Tensor,
        to_sq: torch.Tensor,
    ) -> torch.Tensor:
        """from_sq, to_sq: (B, K) -> logits (B, K)."""
        style = self._ordered_style(hist_w, hist_b, hlen_w, hlen_b, side_to_move)
        turn_e = self.turn_emb(side_to_move.long())
        k = from_sq.shape[1]
        ef = self.emb_from(from_sq)
        et = self.emb_to(to_sq)
        feat = torch.cat(
            [
                cur_emb.unsqueeze(1).expand(-1, k, -1),
                ef,
                et,
                style.unsqueeze(1).expand(-1, k, -1),
                turn_e.unsqueeze(1).expand(-1, k, -1),
            ],
            dim=-1,
        )
        return self.mlp(feat).squeeze(-1)

    def forward(
        self,
        cur_emb: torch.Tensor,
        hist_w: torch.Tensor,
        hist_b: torch.Tensor,
        hlen_w: torch.Tensor,
        hlen_b: torch.Tensor,
        side_to_move: torch.Tensor,
        from_sq: torch.Tensor,
        to_sq: torch.Tensor,
    ) -> torch.Tensor:
        return self.score_moves(
            cur_emb, hist_w, hist_b, hlen_w, hlen_b, side_to_move, from_sq, to_sq
        )
