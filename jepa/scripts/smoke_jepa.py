#!/usr/bin/env python3
"""Minimal smoke: synthetic HDF5, one train step, val forward, checkpoint write."""

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    train_py = _REPO_ROOT / "jepa" / "train.py"
    return subprocess.call([sys.executable, str(train_py), "--smoke"])


if __name__ == "__main__":
    sys.exit(main())
