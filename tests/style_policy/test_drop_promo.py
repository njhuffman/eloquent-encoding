"""Task 1: verify promotion head is gone and legacy checkpoints with promo_head.* load cleanly."""
import torch
from style_policy.model import BasePolicy

_TINY = {"d_model": 32, "n_layers": 1, "nhead": 4, "dim_feedforward": 64, "dropout": 0.0,
         "head_hidden": 32, "elo_dim": 0, "n_elo_buckets": 0}


def test_no_promo_head_attr():
    """BasePolicy built from config must not have a promo_head attribute."""
    m = BasePolicy.from_config(_TINY)
    assert not hasattr(m, "promo_head"), "promo_head attribute should not exist on BasePolicy"


def test_legacy_checkpoint_loads_with_promo_head_keys():
    """A state_dict that contains promo_head.* keys (legacy checkpoint simulation) must load via
    strict=False and return unexpected_keys all starting with 'promo_head'; the model must still
    produce a forward pass."""
    m = BasePolicy.from_config(_TINY)
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    # Inject a fake promo_head entry as a legacy checkpoint would have
    sd["promo_head.x"] = torch.zeros(4, 64)
    sd["promo_head.score.0.weight"] = torch.zeros(8, 64)

    result = m.load_state_dict(sd, strict=False)

    assert all(k.startswith("promo_head") for k in result.unexpected_keys), \
        f"unexpected keys should all be promo_head.*, got {result.unexpected_keys}"
    assert result.missing_keys == [], \
        f"should have no missing keys, got {result.missing_keys}"

    # Model should still do a forward pass after loading
    import numpy as np
    from style_policy.board_encode import board_to_packed
    import chess
    packed = torch.from_numpy(board_to_packed(chess.Board())[None])
    with torch.no_grad():
        cls, squares = m.encode(packed)
    assert cls.shape == (1, 32)
    assert squares.shape == (1, 64, 32)
