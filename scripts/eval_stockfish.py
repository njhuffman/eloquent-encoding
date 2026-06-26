#!/usr/bin/env python3
"""Score dataset positions with Stockfish into a resumable sidecar HDF5.

Run: python -m scripts.eval_stockfish [--h5 ...] [--sample N] [--depth 8] [--workers 8]
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
import h5py
import chess, chess.engine
import multiprocessing as mp
from dataset_generation.stockfish_eval import (
    CP_CLAMP, select_rows, open_or_create_sidecar, pending_positions, write_records,
    StaticEvalEngine, eval_position,
)
from style_policy.board_encode import packed_to_board

_SE = _ST = _DEPTH = None


def _init(sf_path, hash_mb, depth):
    global _SE, _ST, _DEPTH
    _SE = chess.engine.SimpleEngine.popen_uci(sf_path)
    _SE.configure({"Threads": 1, "Hash": hash_mb, "UCI_ShowWDL": True})
    _ST = StaticEvalEngine(sf_path)
    _DEPTH = depth
    import atexit
    atexit.register(_cleanup)


def _cleanup():
    try: _SE.quit()
    except Exception: pass
    try: _ST.close()
    except Exception: pass


def _work(item):
    pos, fen = item
    return pos, eval_position(_SE, _ST, chess.Board(fen), _DEPTH)


def _sf_version(sf_path):
    e = chess.engine.SimpleEngine.popen_uci(sf_path)
    try:
        return e.id.get("name", "unknown")
    finally:
        e.quit()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="/mnt/eloquence_bulk/databases/wdl_validation_1M.h5")
    ap.add_argument("--out", default=None)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard-size", type=int, default=5000)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stockfish", default="/usr/games/stockfish")
    ap.add_argument("--hash", type=int, default=32)
    a = ap.parse_args()
    out = a.out or (os.path.splitext(a.h5)[0] + ".sf_eval.h5")

    with h5py.File(a.h5, "r") as f:
        n_rows = int(f["packed_pre"].shape[0])
        rows = select_rows(n_rows, a.sample, a.seed)
        packed = f["packed_pre"][rows]      # gathered in `rows` order (sorted index list)

    attrs = {"source_h5": a.h5, "source_n_rows": n_rows, "depth": a.depth,
             "sample_n": (a.sample if a.sample is not None else -1), "seed": a.seed,
             "perspective": "STM", "wdl_order": "loss,draw,win", "cp_clamp": CP_CLAMP,
             "stockfish_version": _sf_version(a.stockfish)}
    sc = open_or_create_sidecar(out, rows, attrs)
    try:
        pend = pending_positions(sc)
        print(f"{len(rows)} rows selected, {len(pend)} pending -> {out}", flush=True)
        items = [(int(pos), packed_to_board(np.asarray(packed[pos], np.uint8)).fen()) for pos in pend]
        t0 = time.time()
        buf_pos, buf_rec, n_done = [], [], 0
        with mp.Pool(a.workers, initializer=_init, initargs=(a.stockfish, a.hash, a.depth)) as pool:
            for pos, rec in pool.imap_unordered(_work, items, chunksize=16):
                buf_pos.append(pos); buf_rec.append(rec); n_done += 1
                if len(buf_pos) >= a.shard_size:
                    write_records(sc, buf_pos, buf_rec); buf_pos, buf_rec = [], []
                    print(f"  {n_done}/{len(items)} ({n_done/(time.time()-t0):.0f}/s)", flush=True)
            if buf_pos:
                write_records(sc, buf_pos, buf_rec)
        print(f"done {n_done} in {time.time()-t0:.0f}s", flush=True)
    finally:
        sc.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
