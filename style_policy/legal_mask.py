"""Bitboard (uint64) → (B,64) boolean square mask. Bit s ↔ square s (a1=0 .. h8=63)."""
from __future__ import annotations
import torch


def u64_to_mask(u64: torch.Tensor) -> torch.Tensor:
    bits = torch.arange(64, device=u64.device, dtype=torch.int64)
    vals = u64.to(torch.int64).unsqueeze(-1)  # (B,1)
    return ((vals >> bits) & 1).to(torch.bool)  # (B,64)
