"""Root conftest.py — ensures the project root is on sys.path for pytest."""
import sys
import pathlib

# Ensure the workspace root (this file's directory) is on sys.path so that
# top-level packages (style_policy, dataset_generation, etc.) are importable
# regardless of how pytest is invoked.
_root = str(pathlib.Path(__file__).parent.resolve())
if _root not in sys.path:
    sys.path.insert(0, _root)
