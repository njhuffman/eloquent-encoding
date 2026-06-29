import h5py, numpy as np
from style_policy.packed_codec import PACKED_BOARD_LEN

def write_synth_h5(path, elos, *, seed=0):
    """Tiny WDL-schema h5: empty boards, all-legal masks, given elo_to_move list."""
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
    return path
