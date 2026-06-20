"""
Named model registry under jepa/artifacts: registry.json + per-name checkpoint.pt
"""

from __future__ import annotations

import json
import random
import secrets
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jepa.config import ARTIFACTS_DIR, CHECKPOINT_BASENAME, REGISTRY_FILENAME

_ADJECTIVES = (
    "swift", "calm", "quiet", "bold", "keen", "sharp", "bright", "steady", "nimble", "grand",
    "royal", "iron", "amber", "jade", "crimson", "silver", "golden", "shadow", "frost", "ember",
)
_NOUNS = (
    "rook", "bishop", "knight", "pawn", "castle", "gambit", "fork", "pin", "skewer", "tempo",
    "file", "rank", "square", "check", "mate", "exchange", "endgame", "opening", "blitz", "study",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def artifacts_path(repo_root: Path | None = None, artifacts_dir: Path | str | None = None) -> Path:
    root = repo_root or _repo_root()
    rel = artifacts_dir if artifacts_dir is not None else ARTIFACTS_DIR
    return (root / rel).resolve()


def registry_path(repo_root: Path | None = None, artifacts_dir: Path | str | None = None) -> Path:
    return artifacts_path(repo_root, artifacts_dir) / REGISTRY_FILENAME


def _load_registry_data(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"models": []}
    with open(path) as f:
        return json.load(f)


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(data, indent=2, sort_keys=False)
    fd, tmp = tempfile.mkstemp(suffix=".json", dir=path.parent)
    try:
        with open(fd, "w") as f:
            f.write(raw)
        Path(tmp).replace(path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def list_registered_models(
    repo_root: Path | None = None,
    artifacts_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    return list(_load_registry_data(registry_path(repo_root, artifacts_dir)).get("models", []))


def generate_model_name(
    repo_root: Path | None = None,
    artifacts_dir: Path | str | None = None,
) -> str:
    existing = {m.get("name") for m in list_registered_models(repo_root, artifacts_dir)}
    rng = random.Random()
    for _ in range(10_000):
        base = f"{rng.choice(_ADJECTIVES)}-{rng.choice(_NOUNS)}"
        if base not in existing:
            return base
        suffix = secrets.token_hex(2)
        cand = f"{base}-{suffix}"
        if cand not in existing:
            return cand
    raise RuntimeError("Could not allocate a unique model name")


def register_model(
    *,
    name: str,
    architecture_id: str,
    architecture_config: dict[str, Any],
    train_meta: dict[str, Any],
    train_hparams: dict[str, Any],
    checkpoint_payload: dict[str, Any],
    repo_root: Path | None = None,
    artifacts_dir: Path | str | None = None,
    note: str | None = None,
    training_spec: dict[str, Any] | None = None,
) -> Path:
    root = repo_root or _repo_root()
    art = artifacts_path(root, artifacts_dir)
    reg_file = art / REGISTRY_FILENAME
    data = _load_registry_data(reg_file)
    models: list[dict[str, Any]] = list(data.get("models", []))
    if any(m.get("name") == name for m in models):
        raise ValueError(f"Model name already registered: {name!r}")

    model_dir = art / name
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_dir / CHECKPOINT_BASENAME
    import torch

    torch.save(checkpoint_payload, ckpt_path)

    if training_spec is not None:
        spec_path = model_dir / "training_spec.json"
        _atomic_write_json(spec_path, training_spec)

    try:
        ckpt_relpath = str(ckpt_path.relative_to(root))
    except ValueError:
        ckpt_relpath = str(ckpt_path)

    entry: dict[str, Any] = {
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_relpath": ckpt_relpath,
        "architecture_id": architecture_id,
        "architecture_config": dict(architecture_config),
        "n_train_boards": train_meta.get("n_train_boards"),
        "n_val_boards": train_meta.get("n_val_boards"),
        "train_h5_basename": train_meta.get("train_h5_basename"),
        "val_h5_basename": train_meta.get("val_h5_basename"),
        "train_hparams": dict(train_hparams),
    }
    if note:
        entry["note"] = note
    if training_spec is not None:
        entry["training_spec"] = training_spec
    models.append(entry)
    data["models"] = models
    _atomic_write_json(reg_file, data)
    return ckpt_path


def print_registry_table(
    repo_root: Path | None = None,
    artifacts_dir: Path | str | None = None,
) -> None:
    rows = list_registered_models(repo_root, artifacts_dir)
    if not rows:
        print("No registered models.")
        return
    headers = ("name", "architecture_id", "n_train", "n_val", "best_val_loss", "checkpoint")
    print("\t".join(headers))
    for m in rows:
        th = m.get("train_hparams") or {}
        print(
            "\t".join(
                str(x)
                for x in (
                    m.get("name", ""),
                    m.get("architecture_id", ""),
                    m.get("n_train_boards", ""),
                    m.get("n_val_boards", ""),
                    th.get("best_val_loss", ""),
                    m.get("checkpoint_relpath", ""),
                )
            )
        )
