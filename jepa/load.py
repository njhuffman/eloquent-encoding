"""Load Chess-JEPA modules from registry names or checkpoint files."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from jepa.architectures import DEFAULT_ARCHITECTURE_ID, build_model
from jepa.config import ARTIFACTS_DIR, REGISTRY_FILENAME


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
            "Checkpoint has no architecture_id; assuming chess_jepa_v1 defaults.",
            stacklevel=2,
        )
        arch_id = DEFAULT_ARCHITECTURE_ID
        arch_cfg = {}
    model = build_model(arch_id, arch_cfg)
    model.load_state_dict(state["model_state_dict"], strict=strict)
    return model


def load_jepa_from_checkpoint(
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
    with open(reg_path) as f:
        return json.load(f)


def load_jepa_by_name(
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
        raise KeyError(f"No registered JEPA model named {name!r}. Known: {names}")
    rel = entry.get("checkpoint_relpath")
    if not rel:
        raise KeyError(f"Registry entry for {name!r} has no checkpoint_relpath")
    ckpt = (root / rel).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint for {name!r} not found: {ckpt}")
    return load_jepa_from_checkpoint(ckpt, device=device, strict=strict)
