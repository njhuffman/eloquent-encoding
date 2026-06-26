"""Stockfish evaluation of dataset positions, written to a resumable sidecar HDF5.

All evals are from the side-to-move's perspective (matching the dataset `result` column).
"""
from __future__ import annotations
import re

CP_CLAMP = 32000        # centipawn clamp; a forced mate is stored as ±CP_CLAMP in the cp column
STATIC_NA = -32768      # sentinel: static eval undefined (e.g. side to move is in check)

_EVAL_RE = re.compile(r"(?:NNUE|Final) evaluation:?\s+([+-]?\d+\.\d+)")


def clamp_cp(cp: int) -> int:
    return max(-CP_CLAMP, min(CP_CLAMP, int(cp)))


def parse_static_eval(text: str) -> int | None:
    """Centipawns from a Stockfish `eval` final line, or None if undefined (in check)."""
    m = _EVAL_RE.search(text)
    if m:
        return round(float(m.group(1)) * 100)
    if "none" in text.lower():
        return None
    raise ValueError(f"could not parse static eval from: {text!r}")


def score_to_cp_mate(score) -> tuple[int, int]:
    """python-chess Score (STM-relative) -> (cp, mate). Mate -> cp ±CP_CLAMP; mate is signed UCI moves."""
    m = score.mate()
    if m is not None:
        return (CP_CLAMP if m > 0 else -CP_CLAMP, int(m))
    return (clamp_cp(score.score()), 0)
