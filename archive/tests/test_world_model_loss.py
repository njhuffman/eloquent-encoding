"""Tests for world_model combined loss (patch JEPA/VICReg)."""

from __future__ import annotations

import torch

from world_model.loss import world_model_loss_forward


def test_world_model_loss_forward_shapes() -> None:
    torch.manual_seed(1)
    d = 32
    b = 3
    z_on = torch.randn(b, d)
    patch_on = torch.randn(b, 64, d)
    patch_hat = torch.randn(b, 64, d)
    patch_pos = torch.randn(b, 64, d)
    fl = torch.randn(b, 64)
    fs = torch.randint(0, 64, (b,), dtype=torch.long)
    fm = torch.zeros(b, 64)
    for i in range(b):
        fm[i, fs[i].item()] = 1.0
        fm[i, (fs[i].item() + 1) % 64] = 1.0
    loss, m = world_model_loss_forward(
        z_on,
        patch_on,
        patch_hat,
        patch_pos,
        fl,
        fs,
        fm,
        jepa_patch_weight=0.5,
        from_sq_ce_weight=1.0,
        sq_ce_label_smoothing=0.0,
        vicreg={"inv_coef": 0.1, "var_coef": 0.1, "cov_coef": 0.0, "std_target": 1.0},
        use_amp_cuda=False,
    )
    assert loss.ndim == 0
    assert "loss" in m
    assert "from_sq_ce" in m
    assert "jepa_patch_weighted" in m
    assert "patch_vicreg_cov" in m
    assert "patch_vicreg_inv" in m
