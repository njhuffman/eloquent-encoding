"""
Global From Predictor (gfp): frozen jepa3 board encoder + trainable from-square head.

Import from the repo root (same pattern as ``jepa3``), for example::

    from gfp.model import GlobalFromPredictor
    from gfp.encoder import load_jepa3_encoder_from_checkpoint
    from gfp.dataset import GfpH5Dataset, collate_gfp_batch, make_gfp_loader
    from gfp.h5_io import assert_gfp_h5, gfp_h5_row_count
"""

from __future__ import annotations

from gfp.build_stream import build_gfp_from_recipe
from gfp.dataset import GfpH5Dataset, collate_gfp_batch, make_gfp_loader, sample_row_indices
from gfp.encoder import load_jepa3_encoder_from_checkpoint
from gfp.h5_io import assert_gfp_h5, gfp_h5_row_count
from gfp.model import FromSquareMlpHead, GlobalFromPredictor
from gfp.model_spec import (
    GFP_ARCHITECTURE_ID,
    load_model_spec,
    resolve_training_config_for_stage,
    spec_path_for_model,
)

__all__ = [
    "GFP_ARCHITECTURE_ID",
    "FromSquareMlpHead",
    "GlobalFromPredictor",
    "GfpH5Dataset",
    "assert_gfp_h5",
    "build_gfp_from_recipe",
    "collate_gfp_batch",
    "gfp_h5_row_count",
    "load_jepa3_encoder_from_checkpoint",
    "load_model_spec",
    "make_gfp_loader",
    "resolve_training_config_for_stage",
    "sample_row_indices",
    "spec_path_for_model",
]
