import torch
from style_policy.policy_heads import FromHead, ToHead


def test_from_head_shape():
    h = FromHead(d_model=32, hidden=64)
    squares = torch.randn(5, 64, 32)
    logits = h(squares)
    assert logits.shape == (5, 64)


def test_to_head_depends_on_from_square():
    h = ToHead(d_model=32, hidden=64).eval()
    squares = torch.randn(2, 64, 32)
    with torch.no_grad():
        a = h(squares, torch.tensor([10, 10]))
        b = h(squares, torch.tensor([20, 20]))
    # Different origin squares must yield different to-logits.
    assert not torch.allclose(a, b)


def test_from_head_elo_conditioning_shifts_logits():
    h = FromHead(d_model=32, hidden=64, elo_dim=8, n_elo_buckets=40).eval()
    squares = torch.randn(2, 64, 32)
    with torch.no_grad():
        lo = h(squares, elo_idx=torch.tensor([0, 0]))
        hi = h(squares, elo_idx=torch.tensor([39, 39]))
    assert not torch.allclose(lo, hi)
