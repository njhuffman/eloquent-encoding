"""Stockfish evaluation of dataset positions, written to a resumable sidecar HDF5.

All evals are from the side-to-move's perspective (matching the dataset `result` column).
"""
from __future__ import annotations
import os
import re
import subprocess
import numpy as np
import h5py
import chess

CP_CLAMP = 32000        # centipawn clamp; a forced mate is stored as ±CP_CLAMP in the cp column
_MISSING = object()
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


def select_rows(n_rows: int, sample: int | None, seed: int) -> np.ndarray:
    if sample is None or sample >= n_rows:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=sample, replace=False)).astype(np.int64)


_MATCH_ATTRS = ("source_h5", "source_n_rows", "depth", "sample_n", "seed")


def open_or_create_sidecar(path: str, row_index: np.ndarray, attrs: dict) -> h5py.File:
    """Open an existing sidecar (validating alignment) or create a fresh one sized to row_index."""
    n = len(row_index)
    if os.path.exists(path):
        f = h5py.File(path, "r+")
        if f["row_index"].shape[0] != n or not np.array_equal(f["row_index"][:], row_index):
            f.close()
            raise ValueError("sidecar row_index mismatch — different file/sample/seed; refusing to resume")
        for k in _MATCH_ATTRS:
            cur = f.attrs.get(k, _MISSING)
            if cur is _MISSING or cur != attrs[k]:
                f.close()
                raise ValueError(f"sidecar attr {k} mismatch ({cur!r} != {attrs[k]!r})")
        return f
    f = h5py.File(path, "w")
    f.create_dataset("row_index", data=row_index.astype(np.int64))
    f.create_dataset("sf_static_cp", shape=(n,), dtype="int16", fillvalue=STATIC_NA)
    f.create_dataset("sf_cp", shape=(n,), dtype="int16", fillvalue=0)
    f.create_dataset("sf_mate", shape=(n,), dtype="int8", fillvalue=0)
    f.create_dataset("sf_wdl", shape=(n, 3), dtype="int16", fillvalue=0)
    f.create_dataset("done", shape=(n,), dtype=bool, fillvalue=False)
    for k, v in attrs.items():
        f.attrs[k] = v
    f.flush()
    return f


def pending_positions(f: h5py.File) -> np.ndarray:
    return np.where(~f["done"][:])[0]


def write_records(f: h5py.File, positions, records) -> None:
    for pos, rec in zip(positions, records):
        f["sf_static_cp"][pos] = rec["static_cp"]
        f["sf_cp"][pos] = rec["cp"]
        f["sf_mate"][pos] = rec["mate"]
        f["sf_wdl"][pos] = rec["wdl"]
        f["done"][pos] = True
    f.flush()


class StaticEvalEngine:
    """A raw Stockfish process used only for the static `eval` command (python-chess lacks it)."""
    def __init__(self, sf_path: str):
        self.p = subprocess.Popen([sf_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, bufsize=1)
        self._send("uci")
        for line in self.p.stdout:           # drain until uciok
            if line.strip() == "uciok":
                break

    def _send(self, cmd: str) -> None:
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()

    def eval_cp(self, fen: str) -> int | None:
        self._send(f"position fen {fen}")
        self._send("eval")
        self._send("isready")   # sentinel: drain until readyok so no lines bleed into next call
        result = None
        for line in self.p.stdout:
            s = line.strip()
            if result is None and (s.startswith("NNUE evaluation") or s.startswith("Final evaluation")):
                result = parse_static_eval(s)   # cp (white-relative) or None (in check)
            if s == "readyok":
                break
        return result

    def close(self) -> None:
        try:
            self._send("quit")
            self.p.wait(timeout=2)
        except Exception:
            self.p.kill()


def eval_position(simple_engine, static_engine: StaticEvalEngine, board, depth: int) -> dict:
    """STM-relative record: {cp, mate, static_cp, wdl=[loss,draw,win]}. Terminal board -> sentinels."""
    import chess.engine
    if board.is_game_over():
        return {"cp": 0, "mate": 0, "static_cp": STATIC_NA, "wdl": (0, 0, 0)}
    info = simple_engine.analyse(board, chess.engine.Limit(depth=depth))
    cp, mate = score_to_cp_mate(info["score"].pov(board.turn))
    wdl_info = info.get("wdl")
    if wdl_info is not None:
        w = wdl_info.pov(board.turn)          # Wdl(wins, draws, losses), permille
        wdl = (int(w.losses), int(w.draws), int(w.wins))   # [loss, draw, win]
    else:
        wdl = (0, 0, 0)
    white_cp = static_engine.eval_cp(board.fen())          # white-relative, or None (in check)
    if white_cp is None:
        static_cp = STATIC_NA
    else:
        stm_cp = white_cp if board.turn == chess.WHITE else -white_cp
        static_cp = clamp_cp(stm_cp)
    return {"cp": cp, "mate": mate, "static_cp": static_cp, "wdl": wdl}
