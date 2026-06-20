import math, torch
from style_policy.loss import masked_square_ce, top1_legal


def test_loss_ignores_illegal_and_is_finite():
    logits = torch.zeros(1, 64)
    logits[0, 5] = 100.0  # huge logit on an ILLEGAL square must not affect loss
    mask = torch.zeros(1, 64, dtype=torch.bool)
    mask[0, 0] = True
    mask[0, 1] = True
    target = torch.tensor([0])
    loss = masked_square_ce(logits, target, mask)
    # two legal squares, equal logits → -log(0.5)
    assert math.isfinite(loss.item())
    assert abs(loss.item() - math.log(2)) < 1e-4


def test_top1_among_legal():
    logits = torch.zeros(1, 64)
    logits[0, 1] = 5.0
    mask = torch.zeros(1, 64, dtype=torch.bool)
    mask[0, 0] = mask[0, 1] = True
    assert top1_legal(logits, torch.tensor([1]), mask) == 1.0
    assert top1_legal(logits, torch.tensor([0]), mask) == 0.0


def test_label_smoothing_with_mask_is_finite():
    """Label smoothing must spread mass over LEGAL squares only, not the -inf illegal ones.
    Plain F.cross_entropy(label_smoothing>0) on -inf-masked logits gives inf; this must not."""
    logits = torch.randn(4, 64)
    mask = torch.zeros(4, 64, dtype=torch.bool)
    mask[:, :5] = True  # 5 legal squares
    target = torch.tensor([0, 1, 2, 3])
    loss = masked_square_ce(logits, target, mask, label_smoothing=0.1)
    assert math.isfinite(loss.item()), f"label-smoothed masked loss not finite: {loss.item()}"


def test_label_smoothing_uniform_equals_nll():
    """With uniform logits over the legal set, smoothing leaves the loss unchanged (= -log(1/k))."""
    logits = torch.zeros(1, 64)
    mask = torch.zeros(1, 64, dtype=torch.bool)
    mask[0, :4] = True  # 4 equal legal squares
    target = torch.tensor([0])
    plain = masked_square_ce(logits, target, mask, label_smoothing=0.0).item()
    smoothed = masked_square_ce(logits, target, mask, label_smoothing=0.1).item()
    assert abs(plain - math.log(4)) < 1e-4
    assert abs(smoothed - plain) < 1e-4  # uniform → smoothing is a no-op


def test_all_illegal_row_no_nan():
    """Row 0 valid (2 legal squares, equal logits → log 2); row 1 fully illegal (mask all False).
    Result must be finite and equal to log(2) from the valid row only."""
    logits = torch.zeros(2, 64)
    mask = torch.zeros(2, 64, dtype=torch.bool)
    # Row 0: two legal squares with equal logits
    mask[0, 0] = True
    mask[0, 1] = True
    # Row 1: fully illegal — mask stays all False
    target = torch.tensor([0, 0])
    loss = masked_square_ce(logits, target, mask)
    assert math.isfinite(loss.item()), f"loss is NaN or Inf: {loss.item()}"
    assert abs(loss.item() - math.log(2)) < 1e-4, f"expected log(2)≈{math.log(2):.4f}, got {loss.item():.4f}"
