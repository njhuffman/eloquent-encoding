"""Build HDF5 move-sample datasets from Lichess .pgn.zst month dumps using a YAML recipe."""

from __future__ import annotations

__all__ = ["build_from_recipe", "recipe"]


def __getattr__(name: str):
    if name in ("build_from_recipe", "recipe"):
        from dataset_generation import recipe as _recipe
        from dataset_generation.builder import build_from_recipe as _bfr
        import sys
        _mod = sys.modules[__name__]
        _mod.recipe = _recipe
        _mod.build_from_recipe = _bfr
        if name == "build_from_recipe":
            return _bfr
        return _recipe
    raise AttributeError(f"module 'dataset_generation' has no attribute {name!r}")
