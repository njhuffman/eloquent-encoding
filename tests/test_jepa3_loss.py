"""Tests for jepa3 combined loss."""

from __future__ import annotations

import torch

from jepa3.loss import jepa3_loss_forward, masked_square_ce


def test_masked_square_ce_label_smoothing_partial_mask_finite() -> None:
    """Regression: CE + label_smoothing over partial legal mask must not inf/nan."""
    torch.manual_seed(2)
    b, c = 32, 64
    logits = torch.randn(b, c)
    mask = torch.zeros(b, c)
    mask[:, :20] = 1.0
    labels = torch.randint(0, 20, (b,), dtype=torch.long)
    ce, m = masked_square_ce(logits, labels, mask, label_smoothing=0.1)
    assert ce.isfinite().item()
    assert m["top1"] == m["top1"]


def test_masked_square_ce_ignores_masked_logits() -> None:
    torch.manual_seed(0)
    b, c = 4, 64
    logits = torch.randn(b, c)
    labels = torch.tensor([0, 5, 10, 20], dtype=torch.long)
    mask = torch.zeros(b, c)
    for i in range(b):
        mask[i, labels[i].item()] = 1.0
        if i < 3:
            mask[i, (labels[i].item() + 1) % 64] = 1.0
    ce, m = masked_square_ce(logits, labels, mask, label_smoothing=0.0)
    assert ce.shape == ()
    assert ce.item() == ce.item()
    assert "top1" in m


def test_jepa3_loss_forward_shapes() -> None:
    torch.manual_seed(1)
    d = 32
    p = 16
    b = 3
    z_on = torch.randn(b, d)
    z_hat = torch.randn(b, p)
    z_pos = torch.randn(b, d)
    fl = torch.randn(b, 64)
    tl = torch.randn(b, 64)
    fs = torch.randint(0, 64, (b,), dtype=torch.long)
    ts = torch.randint(0, 64, (b,), dtype=torch.long)
    fm = torch.zeros(b, 64)
    tm = torch.zeros(b, 64)
    for i in range(b):
        fm[i, fs[i].item()] = 1.0
        fm[i, (fs[i].item() + 1) % 64] = 1.0
        tm[i, ts[i].item()] = 1.0
        tm[i, (ts[i].item() + 1) % 64] = 1.0
    loss, m = jepa3_loss_forward(
        z_on,
        z_hat,
        z_pos,
        fl,
        tl,
        fs,
        ts,
        fm,
        tm,
        predictor_prefix_dims=p,
        jepa_weight=1.0,
        from_sq_ce_weight=1.0,
        to_sq_ce_weight=1.0,
        sq_ce_label_smoothing=0.0,
        vicreg={"inv_coef": 0.1, "var_coef": 0.1, "cov_coef": 0.0, "std_target": 1.0},
        use_amp_cuda=False,
    )
    assert loss.ndim == 0
    assert "loss" in m
    assert "from_sq_ce" in m
