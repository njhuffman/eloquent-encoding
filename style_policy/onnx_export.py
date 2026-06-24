"""ONNX-traceable wrappers. They start from a board tensor (B,8,8,18) so the numpy
packed_to_board_tensor step is done outside the graph (in JS). The tricky square-category
logic stays *inside* the encoder graph, guaranteeing parity with training."""
from __future__ import annotations
import torch
import torch.nn as nn


class EncodeExport(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, board_tensor: torch.Tensor):
        cls, squares = self.encoder(board_tensor)
        return squares, cls


class FromHeadExport(nn.Module):
    def __init__(self, from_head: nn.Module):
        super().__init__()
        self.from_head = from_head

    def forward(self, squares: torch.Tensor, elo_idx: torch.Tensor) -> torch.Tensor:
        return self.from_head(squares, elo_idx=elo_idx)


class ToHeadExport(nn.Module):
    def __init__(self, to_head: nn.Module):
        super().__init__()
        self.to_head = to_head

    def forward(self, squares: torch.Tensor, from_sq: torch.Tensor, elo_idx: torch.Tensor) -> torch.Tensor:
        return self.to_head(squares, from_sq, elo_idx=elo_idx)


class ValueHeadExport(nn.Module):
    def __init__(self, value_head: nn.Module):
        super().__init__()
        self.value_head = value_head

    def forward(self, cls: torch.Tensor, elo_idx: torch.Tensor) -> torch.Tensor:
        return self.value_head(cls, elo_idx=elo_idx)


def build_export_modules(policy):
    policy.eval()
    # nn.TransformerEncoder's nested-tensor fast path uses data-dependent ops that don't trace.
    enc = policy.encoder
    if hasattr(enc, "encoder") and hasattr(enc.encoder, "enable_nested_tensor"):
        enc.encoder.enable_nested_tensor = False
    return (EncodeExport(policy.encoder).eval(),
            FromHeadExport(policy.from_head).eval(),
            ToHeadExport(policy.to_head).eval(),
            ValueHeadExport(policy.value_head).eval())
