import math, torch
from style_policy.loss import wdl_ce, wdl_accuracy

def test_wdl_ce_uniform_is_log3():
    logits = torch.zeros(4, 3)  # uniform -> CE = ln(3)
    target = torch.tensor([0, 1, 2, 1])
    assert abs(wdl_ce(logits, target).item() - math.log(3)) < 1e-5

def test_wdl_accuracy():
    logits = torch.tensor([[9.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0], [9.0, 0, 0]])
    target = torch.tensor([0, 1, 2, 2])
    assert abs(wdl_accuracy(logits, target) - 0.75) < 1e-6
