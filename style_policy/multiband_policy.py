"""MultiBandPolicy: shared elo-agnostic encoder + N per-band policy heads + shared value head."""
from __future__ import annotations
import torch
import torch.nn as nn
from style_policy.board_encoder import BoardEncoder
from style_policy.band_head import BandHead
from style_policy.value_head import WDLHead
from style_policy.packed_codec import packed_to_board_tensor

BANDS = list(range(1000, 2000, 100))  # 1000..1900


class MultiBandPolicy(nn.Module):
    def __init__(self, encoder, heads, value_head, bands=BANDS):
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleList(heads)
        self.value_head = value_head
        self.bands = list(bands)
        self.n_bands = len(self.bands)

    @classmethod
    def from_config(cls, cfg: dict) -> "MultiBandPolicy":
        d = int(cfg["d_model"]); h = int(cfg["head_hidden"])
        enc = BoardEncoder(d_model=d, n_layers=int(cfg["n_layers"]), nhead=int(cfg["nhead"]),
                           dim_feedforward=int(cfg["dim_feedforward"]), dropout=float(cfg["dropout"]),
                           use_castling_ep=bool(cfg.get("use_castling_ep", False)),
                           use_last_move=bool(cfg.get("use_last_move", False)),
                           n_history_ply=int(cfg.get("n_history_ply", 4)))
        bands = list(cfg.get("bands", BANDS))
        use_cls = bool(cfg.get("use_cls_in_heads", False))
        heads = [BandHead(d, h, use_cls=use_cls) for _ in bands]
        value = WDLHead(d_model=d, hidden=h, elo_dim=int(cfg.get("elo_dim", 0)),
                        n_elo_buckets=int(cfg.get("n_elo_buckets", 0)))
        return cls(enc, heads, value, bands=bands)

    def encode(self, packed_pre, hist=None):
        board = packed_to_board_tensor(packed_pre).to(next(self.parameters()).device)
        return self.encoder(board, hist=hist)

    def head_index(self, elo: torch.Tensor) -> torch.Tensor:
        """Route each elo to its band head. Bands are 100-wide and contiguous from bands[0];
        elos below the first band map to head 0 and at/above the last band to the final head.
        For the default 10 bands (1000-1900) this is identical to the old clamp(1000,1999)//100."""
        idx = ((elo - self.bands[0]) // 100).long()
        return idx.clamp(0, self.n_bands - 1)
