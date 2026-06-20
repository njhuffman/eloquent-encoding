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
