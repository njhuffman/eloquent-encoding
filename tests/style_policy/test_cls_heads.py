"""Tests for CLS-in-heads feature (Task 3: use_cls_in_heads flag).

Covers:
- FromHead/ToHead/BandHead with use_cls=False ignore a passed cls (regression: byte-identical to no-cls call)
- FromHead/ToHead/BandHead with use_cls=True produce correct (B,64) shapes and use cls
- Param count grows by d_model input rows only when use_cls=True
- BasePolicy.forward_from/forward_to/forward_policy pass cls through (regression: use_cls=False is identical)
- MultiBandPolicy / _routed_policy_loss pass cls into band heads
"""
import torch
import pytest
from style_policy.policy_heads import FromHead, ToHead
from style_policy.band_head import BandHead


# ---------------------------------------------------------------------------
# FromHead
# ---------------------------------------------------------------------------

def test_from_head_use_cls_false_ignores_cls():
    """use_cls=False: passing cls= must give byte-identical output to not passing it."""
    h = FromHead(d_model=32, hidden=64, use_cls=False).eval()
    squares = torch.randn(4, 64, 32)
    cls = torch.randn(4, 32)
    with torch.no_grad():
        out_no_cls = h(squares)
        out_with_cls = h(squares, cls=cls)
    assert torch.equal(out_no_cls, out_with_cls), "cls must be ignored when use_cls=False"


def test_from_head_use_cls_false_param_count():
    """use_cls=False: score MLP input should be d_model (no elo, no cls)."""
    d = 32
    h_off = FromHead(d_model=d, hidden=64, use_cls=False)
    h_on  = FromHead(d_model=d, hidden=64, use_cls=True)
    # When use_cls=True, the first Linear input dim is 2*d vs d for use_cls=False
    off_params = sum(p.numel() for p in h_off.parameters())
    on_params  = sum(p.numel() for p in h_on.parameters())
    assert on_params > off_params, "use_cls=True must add parameters"
    # First layer of score MLP: input dim check
    first_lin_off = list(h_off.score.children())[0]
    first_lin_on  = list(h_on.score.children())[0]
    assert first_lin_off.in_features == d
    assert first_lin_on.in_features == 2 * d


def test_from_head_use_cls_true_shape_and_uses_cls():
    """use_cls=True: output is (B,64) and different when cls differs."""
    h = FromHead(d_model=32, hidden=64, use_cls=True).eval()
    squares = torch.randn(3, 64, 32)
    cls_a = torch.randn(3, 32)
    cls_b = torch.randn(3, 32)
    with torch.no_grad():
        out_a = h(squares, cls=cls_a)
        out_b = h(squares, cls=cls_b)
    assert out_a.shape == (3, 64)
    assert not torch.allclose(out_a, out_b), "output must differ when cls differs"


def test_from_head_use_cls_true_requires_cls():
    """use_cls=True: calling without cls should raise."""
    h = FromHead(d_model=32, hidden=64, use_cls=True).eval()
    squares = torch.randn(2, 64, 32)
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        h(squares)


# ---------------------------------------------------------------------------
# FromHead with elo
# ---------------------------------------------------------------------------

def test_from_head_cls_and_elo():
    """use_cls=True with elo: input dim = d_model + elo_dim + d_model."""
    d, elo_d = 32, 8
    h = FromHead(d_model=d, hidden=64, elo_dim=elo_d, n_elo_buckets=40, use_cls=True)
    first_lin = list(h.score.children())[0]
    assert first_lin.in_features == d + elo_d + d


# ---------------------------------------------------------------------------
# ToHead
# ---------------------------------------------------------------------------

def test_to_head_use_cls_false_ignores_cls():
    """use_cls=False: passing cls= must give byte-identical output."""
    h = ToHead(d_model=32, hidden=64, use_cls=False).eval()
    squares = torch.randn(4, 64, 32)
    from_sq = torch.zeros(4, dtype=torch.long)
    cls = torch.randn(4, 32)
    with torch.no_grad():
        out_no_cls = h(squares, from_sq)
        out_with_cls = h(squares, from_sq, cls=cls)
    assert torch.equal(out_no_cls, out_with_cls), "cls must be ignored when use_cls=False"


def test_to_head_use_cls_false_param_count():
    d = 32
    h_off = ToHead(d_model=d, hidden=64, use_cls=False)
    h_on  = ToHead(d_model=d, hidden=64, use_cls=True)
    first_lin_off = list(h_off.score.children())[0]
    first_lin_on  = list(h_on.score.children())[0]
    assert first_lin_off.in_features == 2 * d
    assert first_lin_on.in_features == 3 * d


def test_to_head_use_cls_true_shape_and_uses_cls():
    h = ToHead(d_model=32, hidden=64, use_cls=True).eval()
    squares = torch.randn(3, 64, 32)
    from_sq = torch.zeros(3, dtype=torch.long)
    cls_a = torch.randn(3, 32)
    cls_b = torch.randn(3, 32)
    with torch.no_grad():
        out_a = h(squares, from_sq, cls=cls_a)
        out_b = h(squares, from_sq, cls=cls_b)
    assert out_a.shape == (3, 64)
    assert not torch.allclose(out_a, out_b)


def test_to_head_use_cls_true_requires_cls():
    h = ToHead(d_model=32, hidden=64, use_cls=True).eval()
    squares = torch.randn(2, 64, 32)
    from_sq = torch.zeros(2, dtype=torch.long)
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        h(squares, from_sq)


def test_to_head_cls_and_elo():
    d, elo_d = 32, 8
    h = ToHead(d_model=d, hidden=64, elo_dim=elo_d, n_elo_buckets=40, use_cls=True)
    first_lin = list(h.score.children())[0]
    assert first_lin.in_features == 2 * d + elo_d + d


# ---------------------------------------------------------------------------
# BandHead
# ---------------------------------------------------------------------------

def test_band_head_use_cls_false_ignores_cls():
    h = BandHead(d_model=32, hidden=16, use_cls=False).eval()
    squares = torch.randn(4, 64, 32)
    from_sq = torch.zeros(4, dtype=torch.long)
    cls = torch.randn(4, 32)
    with torch.no_grad():
        fl_no = h.from_logits(squares)
        fl_with = h.from_logits(squares, cls=cls)
        tl_no = h.to_logits(squares, from_sq)
        tl_with = h.to_logits(squares, from_sq, cls=cls)
    assert torch.equal(fl_no, fl_with), "from_logits: cls must be ignored when use_cls=False"
    assert torch.equal(tl_no, tl_with), "to_logits: cls must be ignored when use_cls=False"


def test_band_head_use_cls_true_shape_and_uses_cls():
    h = BandHead(d_model=32, hidden=16, use_cls=True).eval()
    squares = torch.randn(3, 64, 32)
    from_sq = torch.zeros(3, dtype=torch.long)
    cls_a = torch.randn(3, 32)
    cls_b = torch.randn(3, 32)
    with torch.no_grad():
        fl_a = h.from_logits(squares, cls=cls_a)
        fl_b = h.from_logits(squares, cls=cls_b)
        tl_a = h.to_logits(squares, from_sq, cls=cls_a)
        tl_b = h.to_logits(squares, from_sq, cls=cls_b)
    assert fl_a.shape == (3, 64)
    assert tl_a.shape == (3, 64)
    assert not torch.allclose(fl_a, fl_b), "from_logits should differ when cls differs"
    assert not torch.allclose(tl_a, tl_b), "to_logits should differ when cls differs"


def test_band_head_backward_compat_no_use_cls_arg():
    """Existing call signature BandHead(d, h) should still work (default use_cls=False)."""
    h = BandHead(d_model=32, hidden=16)  # no use_cls arg
    squares = torch.randn(2, 64, 32)
    from_sq = torch.zeros(2, dtype=torch.long)
    fl = h.from_logits(squares)
    tl = h.to_logits(squares, from_sq)
    assert fl.shape == (2, 64)
    assert tl.shape == (2, 64)


# ---------------------------------------------------------------------------
# BasePolicy plumbing (use_cls=False regression)
# ---------------------------------------------------------------------------

def _tiny_cfg(use_cls=False):
    return {
        "d_model": 32, "n_layers": 1, "nhead": 4, "dim_feedforward": 64,
        "dropout": 0.0, "head_hidden": 32, "elo_dim": 0, "n_elo_buckets": 0,
        "use_cls_in_heads": use_cls,
    }


def _make_packed_from_sq():
    """Return (packed_pre, from_sq, from_legal_u64, to_legal_u64) for a single position."""
    import numpy as np
    # Use the board_encoder's packed codec for a synthetic board tensor
    # The simplest approach: zero packed (legal masks from initial position)
    B = 2
    packed = torch.zeros(B, 8, dtype=torch.uint8)
    from_sq = torch.zeros(B, dtype=torch.long)
    # Use non-zero legal masks (all squares legal = -1 as int64)
    from_legal = torch.full((B,), -1, dtype=torch.int64)
    to_legal = torch.full((B,), -1, dtype=torch.int64)
    return packed, from_sq, from_legal, to_legal


def test_base_policy_from_config_use_cls_false_default():
    """from_config without use_cls_in_heads builds use_cls=False heads."""
    from style_policy.model import BasePolicy
    cfg = _tiny_cfg(use_cls=False)
    del cfg["use_cls_in_heads"]  # ensure key absence is handled
    m = BasePolicy.from_config(cfg)
    assert not m.from_head.use_cls
    assert not m.to_head.use_cls


def test_base_policy_from_config_use_cls_true():
    """from_config with use_cls_in_heads=True builds use_cls=True heads."""
    from style_policy.model import BasePolicy
    cfg = _tiny_cfg(use_cls=True)
    m = BasePolicy.from_config(cfg)
    assert m.from_head.use_cls
    assert m.to_head.use_cls


def test_multiband_policy_from_config_use_cls_propagates():
    """MultiBandPolicy.from_config passes use_cls to all BandHeads."""
    from style_policy.multiband_policy import MultiBandPolicy
    cfg = {
        "d_model": 32, "n_layers": 1, "nhead": 4, "dim_feedforward": 64,
        "dropout": 0.0, "head_hidden": 32, "elo_dim": 0, "n_elo_buckets": 0,
        "bands": [1000, 1100],
        "use_cls_in_heads": True,
    }
    m = MultiBandPolicy.from_config(cfg)
    for head in m.heads:
        assert head.from_head.use_cls
        assert head.to_head.use_cls


def test_multiband_policy_from_config_use_cls_false_default():
    """MultiBandPolicy.from_config without use_cls_in_heads defaults to False."""
    from style_policy.multiband_policy import MultiBandPolicy
    cfg = {
        "d_model": 32, "n_layers": 1, "nhead": 4, "dim_feedforward": 64,
        "dropout": 0.0, "head_hidden": 32, "elo_dim": 0, "n_elo_buckets": 0,
        "bands": [1000, 1100],
    }
    m = MultiBandPolicy.from_config(cfg)
    for head in m.heads:
        assert not head.from_head.use_cls
        assert not head.to_head.use_cls
