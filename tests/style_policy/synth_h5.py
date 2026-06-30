import h5py, numpy as np
from style_policy.packed_codec import PACKED_BOARD_LEN

def write_synth_h5(path, elos, *, seed=0, with_hist=False):
    """Tiny WDL-schema h5: empty boards, all-legal masks, given elo_to_move list.

    with_hist=False (default): no hist_* datasets — existing callers unaffected.
    with_hist=True: adds hist_from, hist_to, hist_cap as (N,4) int8 datasets with
        random squares in [0,63] for present plies and -1 sentinels for absent ones.
    """
    n = len(elos)
    rng = np.random.default_rng(seed)
    packed = np.zeros((n, PACKED_BOARD_LEN), dtype=np.uint8)
    packed[:, 33] = 255  # ep = none
    with h5py.File(path, "w") as f:
        f.create_dataset("packed_pre", data=packed)
        f.create_dataset("elo_to_move", data=np.asarray(elos, dtype=np.int16))
        f.create_dataset("opp_elo", data=np.full(n, 1500, dtype=np.int16))
        f.create_dataset("result", data=np.ones(n, dtype=np.int8))
        f.create_dataset("from_sq", data=rng.integers(0, 64, n).astype(np.uint8))
        f.create_dataset("to_sq", data=rng.integers(0, 64, n).astype(np.uint8))
        f.create_dataset("promotion", data=np.zeros(n, dtype=np.uint8))
        f.create_dataset("from_legal_u64", data=np.full(n, np.iinfo(np.uint64).max, dtype=np.uint64))
        f.create_dataset("to_legal_u64", data=np.full(n, np.iinfo(np.uint64).max, dtype=np.uint64))
        if with_hist:
            # Generate random squares in [0,63] for all 4 plies; mark the last ply
            # absent (-1) for half the rows to exercise the sentinel path.
            hist_from = rng.integers(0, 64, (n, 4)).astype(np.int8)
            hist_to   = rng.integers(0, 64, (n, 4)).astype(np.int8)
            hist_cap  = rng.integers(0, 6,  (n, 4)).astype(np.int8)
            # Set ply index 3 (oldest) to absent for every other row
            absent_rows = np.arange(0, n, 2)
            hist_from[absent_rows, 3] = -1
            hist_to[absent_rows, 3]   = -1
            hist_cap[absent_rows, 3]  = 0
            f.create_dataset("hist_from", data=hist_from)
            f.create_dataset("hist_to",   data=hist_to)
            f.create_dataset("hist_cap",  data=hist_cap)
    return path
