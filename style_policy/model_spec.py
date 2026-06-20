"""YAML spec: defaults + per-stage deep-merge; elo→bucket mapping. Paths resolve from repo root."""
from __future__ import annotations
from pathlib import Path
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = _REPO_ROOT / "style_policy" / "model_configs"


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


def load_spec(name: str) -> dict:
    path = CONFIGS / f"{name}.yaml"
    spec = yaml.safe_load(path.read_text())
    if spec.get("name") != name:
        raise ValueError(f"spec name {spec.get('name')!r} != {name!r}")
    defaults = spec.get("defaults", {})
    spec["stages"] = [_deep_merge(defaults, s) for s in spec["stages"]]
    return spec


def elo_to_bucket(elo: torch.Tensor, n_buckets: int) -> torch.Tensor:
    e = elo.long()
    bucket = torch.div(e, 100, rounding_mode="floor").clamp(0, n_buckets - 1)
    return torch.where(e > 0, bucket, torch.full_like(e, n_buckets))
