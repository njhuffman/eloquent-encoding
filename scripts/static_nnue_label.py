"""Label an h5's positions with a Stockfish DEPTH-1 search eval into a row-aligned sidecar.

Why depth-1 and not the static `eval` command: `eval` is a verbose diagnostic that recomputes the
full per-piece NNUE contribution table (~40 evals) per call -> ~10.8ms. A depth-1 search is setup +
a handful of cheap incremental NNUE evals -> ~1.6ms (6x faster), and is a slightly better target
(resolves the immediate tactic instead of scoring as if quiet). 32M in ~1h @ 14 workers.

Sidecar: {row_index, sf_cp (int16, STM-relative cp, clamped to +-CP_CLAMP)}. attrs: search_depth.
NA sentinel (STATIC_NA) only on analyse failure; depth-1 evaluates in-check positions normally."""
from __future__ import annotations
import argparse, os, time, h5py, numpy as np, multiprocessing as mp, atexit
import sys
import chess, chess.engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_generation.stockfish_eval import STATIC_NA, CP_CLAMP
from style_policy.board_encode import packed_to_board

_ENG = None
_DEPTH = 1
_PACKED = None                                             # loaded to RAM in main; workers inherit COW
def _init(sf_path, depth):
    global _ENG, _DEPTH
    _DEPTH = depth
    _ENG = chess.engine.SimpleEngine.popen_uci(sf_path)
    _ENG.configure({"Threads": 1})                         # one core per worker; parallelism is across workers
    atexit.register(lambda: _ENG.quit())

def _work(pos):
    board = packed_to_board(_PACKED[pos])                  # read from shared RAM, no per-worker h5 I/O
    try:
        info = _ENG.analyse(board, chess.engine.Limit(depth=_DEPTH))
        cp = info["score"].pov(board.turn).score(mate_score=CP_CLAMP)   # STM-relative cp (mate -> +-CP_CLAMP)
    except Exception:
        return pos, STATIC_NA
    if cp is None:
        return pos, STATIC_NA
    return pos, int(max(-CP_CLAMP, min(CP_CLAMP, cp)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="/mnt/eloquence_bulk/databases/ourteacher_distill_labels.h5")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--stockfish", default="/usr/games/stockfish")
    a = ap.parse_args()
    out = a.out or a.h5.replace(".h5", ".nnue.h5")
    global _PACKED
    print(f"loading packed_pre to RAM ...", flush=True)
    with h5py.File(a.h5, "r") as f:
        N = int(f["packed_pre"].shape[0])
        _PACKED = f["packed_pre"][:]            # ~1GB; forked workers inherit it COW (no per-worker h5 I/O)
    print(f"labeling {N:,} positions with depth-{a.depth} SF eval ({a.workers} workers) -> {out}", flush=True)
    res = np.full(N, STATIC_NA, dtype=np.int16)
    t0 = time.time(); done = 0
    with mp.Pool(a.workers, _init, (a.stockfish, a.depth)) as p:
        for pos, cp in p.imap_unordered(_work, range(N), chunksize=1024):
            res[pos] = cp; done += 1
            if done % 500000 == 0:
                r = done / (time.time() - t0)
                print(f"  {done:,}/{N:,} ({r:.0f}/s, eta {(N-done)/r/60:.0f}min)", flush=True)
    with h5py.File(out, "w") as o:
        o.create_dataset("row_index", data=np.arange(N, dtype=np.int64))
        o.create_dataset("sf_cp", data=res)
        o.attrs["source_h5"] = a.h5; o.attrs["static_na"] = STATIC_NA
        o.attrs["cp_clamp"] = CP_CLAMP; o.attrs["search_depth"] = a.depth
    valid = int((res != STATIC_NA).sum())
    print(f"done {N:,} in {(time.time()-t0)/60:.0f}min | valid {valid:,} | saved {out}", flush=True)

if __name__ == "__main__":
    main()
