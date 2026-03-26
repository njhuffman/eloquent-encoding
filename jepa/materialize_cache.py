"""
Content-addressed cache for JEPA materialized HDF5s (mining / negatives).

Cache keys include sampling seeds, architecture, move-file fingerprints, and
(when hard mining) the previous-stage checkpoint fingerprint — not LR/epochs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py

MATERIALIZE_CACHE_SCHEMA = 2


def file_fingerprint(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "st_size": int(st.st_size),
        "st_mtime_ns": int(st.st_mtime_ns),
    }


def _canonical_key_json(key: dict[str, Any]) -> str:
    return json.dumps(key, sort_keys=True, separators=(",", ":"))


def hash_materialize_cache_key(key: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_key_json(key).encode("utf-8")).hexdigest()


def materialize_cache_paths(cache_dir: Path, name: str, stage: int, key_hex: str) -> tuple[Path, Path, Path]:
    short = key_hex[:12]
    stem = f"{name}_stage{stage}_{short}"
    train_h5 = cache_dir / f"{stem}_train.h5"
    val_h5 = cache_dir / f"{stem}_val.h5"
    manifest = cache_dir / f"{stem}.cache.json"
    return train_h5, val_h5, manifest


def build_materialize_cache_key(
    *,
    stage: int,
    use_hard_mining: bool,
    architecture_id: str,
    architecture_resolved: dict[str, Any],
    k_neg: int,
    n_hard: int,
    m_random: int,
    mining_position_batch: int,
    train_neg_seed: int,
    val_neg_seed: int,
    n_train_rows: int,
    n_val_rows: int,
    sample_n: int,
    sample_seed: int,
    val_sample_n: int,
    val_sample_seed: int,
    train_move_fingerprint: dict[str, Any],
    val_move_fingerprint: dict[str, Any],
    mining_checkpoint_fingerprint: dict[str, Any] | None,
    evaluate_legals_n: int | None,
) -> dict[str, Any]:
    return {
        "architecture_id": architecture_id,
        "architecture_resolved": dict(architecture_resolved),
        "evaluate_legals_n": evaluate_legals_n,
        "k_neg": int(k_neg),
        "materialize_cache_schema": MATERIALIZE_CACHE_SCHEMA,
        "mining_checkpoint": mining_checkpoint_fingerprint,
        "mining_position_batch": int(mining_position_batch),
        "m_random": int(m_random),
        "n_hard": int(n_hard),
        "n_train_rows": int(n_train_rows),
        "n_val_rows": int(n_val_rows),
        "sample_n": int(sample_n),
        "sample_seed": int(sample_seed),
        "stage": int(stage),
        "train_move": dict(train_move_fingerprint),
        "train_neg_seed": int(train_neg_seed),
        "use_hard_mining": bool(use_hard_mining),
        "val_move": dict(val_move_fingerprint),
        "val_neg_seed": int(val_neg_seed),
        "val_sample_n": int(val_sample_n),
        "val_sample_seed": int(val_sample_seed),
    }


def _report_path_for_h5(h5_path: Path) -> Path:
    return h5_path.with_suffix(".report.json")


def _h5_k_neg_attr(path: Path) -> int | None:
    try:
        with h5py.File(path, "r") as f:
            return int(f.attrs["num_negatives_k"])
    except OSError:
        return None


def try_load_materialize_cache(
    *,
    cache_key: dict[str, Any],
    key_sha256: str,
    train_h5_path: Path,
    val_h5_path: Path,
    manifest_path: Path,
    k_neg: int,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not manifest_path.is_file():
        return None
    if not train_h5_path.is_file() or not val_h5_path.is_file():
        return None
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    if int(raw.get("materialize_cache_schema", -1)) != MATERIALIZE_CACHE_SCHEMA:
        return None
    if raw.get("key_sha256") != key_sha256:
        return None
    if raw.get("key") != cache_key:
        return None
    if Path(raw.get("train_h5", "")) != train_h5_path or Path(raw.get("val_h5", "")) != val_h5_path:
        return None
    if _h5_k_neg_attr(train_h5_path) != k_neg or _h5_k_neg_attr(val_h5_path) != k_neg:
        return None
    tr_path = _report_path_for_h5(train_h5_path)
    va_path = _report_path_for_h5(val_h5_path)
    if not tr_path.is_file() or not va_path.is_file():
        return None
    try:
        rep_tr = json.loads(tr_path.read_text(encoding="utf-8"))
        rep_va = json.loads(va_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(rep_tr, dict) or not isinstance(rep_va, dict):
        return None
    if int(rep_tr.get("n_written", 0)) < 1 or int(rep_va.get("n_written", 0)) < 1:
        return None
    return rep_tr, rep_va


def write_materialize_cache_manifest(
    manifest_path: Path,
    *,
    cache_key: dict[str, Any],
    key_sha256: str,
    train_h5_path: Path,
    val_h5_path: Path,
) -> None:
    payload = {
        "materialize_cache_schema": MATERIALIZE_CACHE_SCHEMA,
        "key": cache_key,
        "key_sha256": key_sha256,
        "train_h5": str(train_h5_path),
        "val_h5": str(val_h5_path),
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
