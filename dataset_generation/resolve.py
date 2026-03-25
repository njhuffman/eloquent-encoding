from __future__ import annotations

from pathlib import Path


def resolve_source_file(data_dir: Path, source: str) -> Path:
    """
    Resolve `source` to a regular file under `data_dir`.
    `source` must be relative (no absolute path, no ..).
    """
    if not source or not str(source).strip():
        raise ValueError("source must be non-empty")
    rel = Path(source)
    if rel.is_absolute():
        raise ValueError(f"source must be relative to data-dir, got absolute: {source!r}")
    if ".." in rel.parts:
        raise ValueError(f"source must not contain '..': {source!r}")

    root = data_dir.expanduser().resolve()
    path = (root / rel).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"source resolves outside data-dir: {source!r}")
    if not path.is_file():
        raise FileNotFoundError(f"PGN.zst not found: {path}")
    return path
