import torch
from style_policy.legal_mask import u64_to_mask


def test_single_bit():
    m = u64_to_mask(torch.tensor([1 << 5], dtype=torch.int64))
    assert m.shape == (1, 64)
    assert m[0, 5].item() is True
    assert m[0].sum().item() == 1


def test_multi_bits():
    val = (1 << 0) | (1 << 63)
    m = u64_to_mask(torch.tensor([val], dtype=torch.uint64))
    assert m[0, 0] and m[0, 63] and m[0].sum().item() == 2
