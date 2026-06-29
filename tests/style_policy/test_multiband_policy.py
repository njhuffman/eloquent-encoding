import torch
from style_policy.multiband_policy import MultiBandPolicy
from style_policy.packed_codec import PACKED_BOARD_LEN

ARCH = {"d_model": 32, "n_layers": 1, "nhead": 4, "dim_feedforward": 64,
        "dropout": 0.0, "head_hidden": 16, "elo_dim": 8, "n_elo_buckets": 40}

def test_head_index_mapping():
    elo = torch.tensor([950, 1000, 1099, 1100, 1500, 1900, 1999, 2050])
    idx = MultiBandPolicy.head_index(elo)
    assert idx.tolist() == [0, 0, 0, 1, 5, 9, 9, 9]

def test_build_and_forward():
    m = MultiBandPolicy.from_config(ARCH)
    assert m.n_bands == 10 and len(m.heads) == 10
    import numpy as np
    packed = torch.zeros(4, PACKED_BOARD_LEN, dtype=torch.uint8); packed[:, 33] = 255
    cls, squares = m.encode(packed)
    assert cls.shape == (4, 32) and squares.shape == (4, 64, 32)
    fl = m.heads[3].from_logits(squares)
    assert fl.shape == (4, 64)
    v = m.value_head(cls, elo_idx=torch.full((4,), 15, dtype=torch.long))
    assert v.shape == (4, 3)
