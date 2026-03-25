"""Build HDF5 move-sample datasets from Lichess .pgn.zst month dumps using a YAML recipe."""

from __future__ import annotations

from dataset_generation import recipe
from dataset_generation.builder import build_from_recipe

__all__ = ["build_from_recipe", "recipe"]
