"""Checkpoint dict layout for Chess-JEPA (mirrors embedding.checkpoint_utils)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import torch.nn as nn


def strip_compile_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    prefix = "_orig_mod."
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    return {k.removeprefix(prefix): v for k, v in state_dict.items()}


def build_model_checkpoint(
    model: nn.Module,
    *,
    architecture_id: str,
    architecture_config: dict[str, Any],
    train_meta: dict[str, Any],
    train_hparams: dict[str, Any],
    optimizer_state_dict: dict[str, Any] | None = None,
    epoch: int | None = None,
    val_loss: float | None = None,
    training_spec: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hparams = dict(train_hparams)
    if epoch is not None:
        hparams.setdefault("epoch", epoch)
    if val_loss is not None:
        hparams.setdefault("best_val_loss", val_loss)
    hparams.setdefault("created_at", datetime.now(timezone.utc).isoformat())

    out: dict[str, Any] = {
        "model_state_dict": strip_compile_prefix(model.state_dict()),
        "architecture_id": architecture_id,
        "architecture_config": dict(architecture_config),
        "train_meta": dict(train_meta),
        "train_hparams": hparams,
        "epoch": hparams.get("epoch"),
        "val_loss": hparams.get("best_val_loss"),
    }
    if optimizer_state_dict is not None:
        out["optimizer_state_dict"] = optimizer_state_dict
    if training_spec is not None:
        out["training_spec"] = training_spec
    if extra:
        out.update(extra)
    return out
