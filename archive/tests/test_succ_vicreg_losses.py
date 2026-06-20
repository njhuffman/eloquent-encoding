"""Tests for legal-successor VICReg loss (jepa2.loss.succ_vicreg_losses)."""

from __future__ import annotations

import unittest

import torch

from jepa2.loss import succ_vicreg_losses


class SuccVicregTests(unittest.TestCase):
    def test_succ_vicreg_higher_spread_higher_var_penalty(self) -> None:
        b, m, d = 2, 4, 8
        mask = torch.ones(b, m)
        # Same embedding for every legal -> per-dim std across M is 0 -> penalty (0 - std_target)^2.
        z_flat = torch.zeros(b, m, d)
        # Large spread across legal index m (same for every dim) -> std across M far from std_target=1.
        idx = torch.arange(m, dtype=torch.float32).view(1, m, 1).expand(b, m, d)
        z_spread = idx * 100.0
        std_target = 1.0
        v_flat, _, _ = succ_vicreg_losses(z_flat, mask, std_target=std_target)
        v_spread, _, _ = succ_vicreg_losses(z_spread, mask, std_target=std_target)
        self.assertGreater(float(v_spread), float(v_flat))

    def test_succ_vicreg_cov_zero_when_d1(self) -> None:
        b, m = 2, 3
        z = torch.randn(b, m, 1)
        mask = torch.ones(b, m)
        _, cov, meta = succ_vicreg_losses(z, mask, std_target=1.0)
        self.assertEqual(cov.item(), 0.0)
        self.assertEqual(meta["succ_vicreg_cov"], 0.0)

    def test_succ_vicreg_skips_row_with_one_legal(self) -> None:
        """Only one legal: row skipped; second row has 3 legals so batch still runs."""
        z = torch.zeros(2, 4, 4)
        z[1] = torch.randn(4, 4)
        mask = torch.tensor([[1.0, 0, 0, 0], [1, 1, 1, 0]])
        var_raw, cov_raw, meta = succ_vicreg_losses(z, mask, std_target=1.0)
        self.assertTrue(torch.isfinite(var_raw))
        self.assertTrue(torch.isfinite(cov_raw))
        self.assertIn("succ_vicreg_var", meta)


if __name__ == "__main__":
    unittest.main()
