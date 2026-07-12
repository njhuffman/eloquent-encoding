"""Label an h5's positions with a Stockfish depth-D search: EVAL + BEST MOVE, in one search per
position (analyse returns both score and pv[0]). Row-aligned sidecar for the multi-task Pass-2 run.

Fields (row-aligned to the source h5):
  sf_cp        int16   STM-relative cp, clamped +-CP_CLAMP  (NA = STATIC_NA on failure)
  sf_best_from int8    best move's from-square (0..63, ABSOLUTE frame; matches human from_sq)
  sf_best_to   int8    best move's to-square   (0..63, ABSOLUTE)
  sf_best_promo int8   promotion piece (chess: 2=N..5=Q, 0=none)  (-1 = no move / failure)
attrs: search_depth. Best move is ABSOLUTE-frame to match the stored human move labels (from_sq,
to_sq); the eval is STM-relative like Pass-1."""
from __future__ import annotations
import argparse, os, time, h5py, numpy as np, multiprocessing as mp, atexit, sys
import chess, chess.engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_generation.stockfish_eval import STATIC_NA, CP_CLAMP
from style_policy.board_encode import packed_to_board

_ENG = None
_DEPTH = 8
_PACKED = None                                             # loaded to RAM in main; workers inherit COW
_NO_MOVE = -1
def _init(sf_path, depth):
    global _ENG, _DEPTH
    _DEPTH = depth
    _ENG = chess.engine.SimpleEngine.popen_uci(sf_path)
    _ENG.configure({"Threads": 1})
    atexit.register(lambda: _ENG.quit())

def _work(pos):
    board = packed_to_board(_PACKED[pos])
    try:
        info = _ENG.analyse(board, chess.engine.Limit(depth=_DEPTH))
        cp = info["score"].pov(board.turn).score(mate_score=CP_CLAMP)
        pv = info.get("pv") or []
        mv = pv[0] if pv else None
    except Exception:
        return pos, STATIC_NA, _NO_MOVE, _NO_MOVE, _NO_MOVE
    cp = STATIC_NA if cp is None else int(max(-CP_CLAMP, min(CP_CLAMP, cp)))
    if mv is None:
        return pos, cp, _NO_MOVE, _NO_MOVE, _NO_MOVE
    return pos, cp, int(mv.from_square), int(mv.to_square), int(mv.promotion or 0)

def _flush(out, cp, bf, bt, bp, dmask, src, depth):
    """Atomic checkpoint: write a full sidecar to out.tmp then rename over out, so the live file
    is ALWAYS a complete, valid checkpoint even if killed mid-write."""
    tmp = out + ".tmp"; N = len(cp)
    with h5py.File(tmp, "w") as o:
        o.create_dataset("row_index", data=np.arange(N, dtype=np.int64))
        o.create_dataset("sf_cp", data=cp)
        o.create_dataset("sf_best_from", data=bf); o.create_dataset("sf_best_to", data=bt)
        o.create_dataset("sf_best_promo", data=bp); o.create_dataset("done", data=dmask)
        o.attrs["source_h5"] = src; o.attrs["static_na"] = STATIC_NA
        o.attrs["cp_clamp"] = CP_CLAMP; o.attrs["search_depth"] = depth
    os.replace(tmp, out)                                    # atomic on same filesystem

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="/mnt/eloquence_bulk/databases/wdl_history_128M.h5")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)         # >0: label only first N rows (benchmark)
    ap.add_argument("--flush-secs", type=int, default=600)  # checkpoint cadence -> bounds work lost on kill
    ap.add_argument("--stockfish", default="/usr/games/stockfish")
    a = ap.parse_args()
    out = a.out or a.h5.replace(".h5", ".sflabels.h5")
    global _PACKED
    print("loading packed_pre to RAM ...", flush=True)
    with h5py.File(a.h5, "r") as f:
        N = int(f["packed_pre"].shape[0])
        if a.limit > 0: N = min(N, a.limit)
        _PACKED = f["packed_pre"][:N]

    # Resume if a partial sidecar exists; else create full-size datasets + a done-mask.
    if os.path.exists(out):
        with h5py.File(out, "r") as o:
            assert o["sf_cp"].shape[0] == N, f"existing {out} has {o['sf_cp'].shape[0]} rows != {N}"
            if int(o.attrs.get("search_depth", a.depth)) != a.depth:
                print(f"WARNING: existing depth {int(o.attrs['search_depth'])} != --depth {a.depth}", flush=True)
            cp = o["sf_cp"][:]; bf = o["sf_best_from"][:]; bt = o["sf_best_to"][:]
            bp = o["sf_best_promo"][:]; dmask = o["done"][:]
        todo = np.nonzero(dmask == 0)[0]
        print(f"RESUMING {out}: {int((dmask != 0).sum()):,}/{N:,} done, {len(todo):,} remaining", flush=True)
    else:
        cp = np.full(N, STATIC_NA, dtype=np.int16)
        bf = np.full(N, _NO_MOVE, dtype=np.int8); bt = np.full(N, _NO_MOVE, dtype=np.int8)
        bp = np.full(N, _NO_MOVE, dtype=np.int8); dmask = np.zeros(N, dtype=np.uint8)
        todo = np.arange(N)
        print(f"labeling {N:,} positions with depth-{a.depth} SF eval+bestmove ({a.workers} workers) -> {out}", flush=True)

    t0 = time.time(); done = 0; last = time.time(); base = int((dmask != 0).sum())
    with mp.Pool(a.workers, _init, (a.stockfish, a.depth)) as p:
        for pos, c, ff, tt, pp in p.imap_unordered(_work, todo, chunksize=512):
            cp[pos] = c; bf[pos] = ff; bt[pos] = tt; bp[pos] = pp; dmask[pos] = 1; done += 1
            if time.time() - last >= a.flush_secs:
                _flush(out, cp, bf, bt, bp, dmask, a.h5, a.depth); last = time.time()
                r = done / (time.time() - t0)
                print(f"  {base+done:,}/{N:,} ({r:.0f}/s, eta {(len(todo)-done)/r/3600:.1f}h) [checkpointed]", flush=True)
    _flush(out, cp, bf, bt, bp, dmask, a.h5, a.depth)
    valid = int((cp != STATIC_NA).sum()); havemv = int((bf != _NO_MOVE).sum())
    print(f"done {N:,} in {(time.time()-t0)/3600:.2f}h | eval_valid {valid:,} | move_valid {havemv:,} | saved {out}", flush=True)

if __name__ == "__main__":
    main()
