"""Label an h5's positions with Stockfish's STATIC NNUE eval (no search) into a row-aligned sidecar.
Fast (~7-10k/s @ 12-16 workers): 32M in ~1h. Sidecar: {row_index, sf_static_cp (int16, STM cp)}."""
from __future__ import annotations
import argparse, os, time, h5py, numpy as np, multiprocessing as mp, atexit
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_generation.stockfish_eval import StaticEvalEngine, STATIC_NA, CP_CLAMP
from style_policy.board_encode import packed_to_board

_ST = _F = None
def _init(sf_path, h5_path):
    global _ST, _F
    _ST = StaticEvalEngine(sf_path); _F = h5py.File(h5_path, "r")
    atexit.register(lambda: _ST.close())

def _work(pos):
    packed = _F["packed_pre"][pos].astype(np.uint8)
    board = packed_to_board(packed)
    cp = _ST.eval_cp(board.fen())                          # static NNUE cp, white-relative (None in check)
    if cp is None:
        return pos, STATIC_NA
    if board.turn == __import__("chess").BLACK:
        cp = -cp                                            # -> side-to-move perspective
    return pos, int(max(-CP_CLAMP, min(CP_CLAMP, cp)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="/mnt/eloquence_bulk/databases/ourteacher_distill_labels.h5")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--stockfish", default="/usr/games/stockfish")
    a = ap.parse_args()
    out = a.out or a.h5.replace(".h5", ".nnue.h5")
    with h5py.File(a.h5, "r") as f:
        N = int(f["packed_pre"].shape[0])
    print(f"labeling {N:,} positions with static NNUE ({a.workers} workers) -> {out}", flush=True)
    res = np.full(N, STATIC_NA, dtype=np.int16)
    t0 = time.time(); done = 0
    with mp.Pool(a.workers, _init, (a.stockfish, a.h5)) as p:
        for pos, cp in p.imap_unordered(_work, range(N), chunksize=1024):
            res[pos] = cp; done += 1
            if done % 500000 == 0:
                r = done / (time.time() - t0)
                print(f"  {done:,}/{N:,} ({r:.0f}/s, eta {(N-done)/r/60:.0f}min)", flush=True)
    with h5py.File(out, "w") as o:
        o.create_dataset("row_index", data=np.arange(N, dtype=np.int64))
        o.create_dataset("sf_static_cp", data=res)
        o.attrs["source_h5"] = a.h5; o.attrs["static_na"] = STATIC_NA; o.attrs["cp_clamp"] = CP_CLAMP
    valid = int((res != STATIC_NA).sum())
    print(f"done {N:,} in {(time.time()-t0)/60:.0f}min | valid {valid:,} | saved {out}", flush=True)

if __name__ == "__main__":
    main()
