#!/usr/bin/env python3
"""Print registered embedding models (see embedding/registry.py)."""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from embedding.config import ARTIFACTS_DIR
from embedding.registry import print_registry_table


def main() -> int:
    p = argparse.ArgumentParser(description="List registered MAE models")
    p.add_argument(
        "--artifacts-dir",
        type=str,
        default=ARTIFACTS_DIR,
        help=f"Registry root (default {ARTIFACTS_DIR})",
    )
    args = p.parse_args()
    print_registry_table(repo_root=_REPO_ROOT, artifacts_dir=args.artifacts_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
