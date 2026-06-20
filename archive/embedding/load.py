"""Load MAE modules from registry names or checkpoint files."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from embedding.architectures import DEFAULT_ARCHITECTURE_ID, build_model, resolve_config_for_id
from embedding.config import ARTIFACTS_DIR, REGISTRY_FILENAME

_ENV_DEFAULT_NAME = "ELOQUENCE_EMBEDDING"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_checkpoint_mapping(
    path: str | Path,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    if map_location is None:
        map_location = "cpu"
    return torch.load(path, map_location=map_location, weights_only=False)


def model_from_checkpoint_state(
    state: dict[str, Any],
    *,
    map_location: str | torch.device | None = None,
    strict: bool = True,
) -> nn.Module:
    if "model_state_dict" not in state:
        raise KeyError("Checkpoint missing model_state_dict")
    if "architecture_id" in state:
        arch_id = state["architecture_id"]
        arch_cfg = state.get("architecture_config") or {}
    else:
        warnings.warn(
            "Checkpoint has no architecture_id; assuming residual_chess_mae_v1 defaults (legacy).",
            stacklevel=2,
        )
        arch_id = DEFAULT_ARCHITECTURE_ID
        arch_cfg = {}
    model = build_model(arch_id, arch_cfg)
    sd = state["model_state_dict"]
    if map_location is not None and isinstance(map_location, str):
        # weights may already be on correct device from torch.load
        pass
    model.load_state_dict(sd, strict=strict)
    return model


def load_mae_from_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device | None = None,
    strict: bool = True,
) -> nn.Module:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = load_checkpoint_mapping(path, map_location=dev)
    model = model_from_checkpoint_state(state, map_location=dev, strict=strict)
    return model.to(dev)


def _read_registry(artifacts_dir: Path) -> dict[str, Any]:
    reg_path = artifacts_dir / REGISTRY_FILENAME
    if not reg_path.is_file():
        return {"models": []}
    import json

    with open(reg_path) as f:
        return json.load(f)


def load_mae_by_name(
    name: str,
    *,
    repo_root: Path | None = None,
    artifacts_dir: Path | None = None,
    device: str | torch.device | None = None,
    strict: bool = True,
) -> nn.Module:
    root = repo_root or _repo_root()
    art = artifacts_dir or (root / ARTIFACTS_DIR)
    data = _read_registry(art)
    models = data.get("models", [])
    entry = next((m for m in models if m.get("name") == name), None)
    if entry is None:
        names = [m.get("name") for m in models]
        raise KeyError(f"No registered model named {name!r}. Known: {names}")
    rel = entry.get("checkpoint_relpath")
    if not rel:
        raise KeyError(f"Registry entry for {name!r} has no checkpoint_relpath")
    ckpt = (root / rel).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint for {name!r} not found: {ckpt}")
    return load_mae_from_checkpoint(ckpt, device=device, strict=strict)


def default_embedding_name() -> str | None:
    v = os.environ.get(_ENV_DEFAULT_NAME)
    return v.strip() if v else None


def load_mae_default_or_name(
    name: str | None = None,
    *,
    repo_root: Path | None = None,
    artifacts_dir: Path | None = None,
    device: str | torch.device | None = None,
    strict: bool = True,
) -> nn.Module:
    n = name or default_embedding_name()
    if not n:
        raise ValueError(
            f"No model name given and {_ENV_DEFAULT_NAME} is unset. "
            "Pass a name or set the environment variable."
        )
    return load_mae_by_name(n, repo_root=repo_root, artifacts_dir=artifacts_dir, device=device, strict=strict)


def resolved_architecture_from_checkpoint(path: str | Path) -> tuple[str, dict[str, Any]]:
    """Return (architecture_id, resolved architecture_config) for a checkpoint file."""
    state = load_checkpoint_mapping(path, map_location="cpu")
    if "architecture_id" in state:
        arch_id = state["architecture_id"]
        partial = state.get("architecture_config") or {}
    else:
        arch_id = DEFAULT_ARCHITECTURE_ID
        partial = {}
    full = resolve_config_for_id(arch_id, partial)
    return arch_id, full
