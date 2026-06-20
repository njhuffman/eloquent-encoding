"""Tests for jepa2.gsnr_probe."""

from __future__ import annotations

import math
import unittest

import torch

from jepa2.gsnr_probe import gsnr_metrics_from_grad_vectors


class GsnrProbeTests(unittest.TestCase):
    def test_gsnr_high_when_grads_aligned(self) -> None:
        # Same direction, different magnitudes -> small noise, positive signal
        u = torch.tensor([1.0, 0.0, 0.0])
        g0 = 1.0 * u
        g1 = 1.1 * u
        g2 = 0.9 * u
        m = gsnr_metrics_from_grad_vectors([g0, g1, g2])
        self.assertTrue(math.isfinite(m["gsnr_encoder"]))
        self.assertGreater(m["gsnr_encoder"], 10.0)

    def test_gsnr_low_when_grads_opposed(self) -> None:
        g0 = torch.tensor([1.0, 0.0])
        g1 = torch.tensor([-1.0, 0.0])
        m = gsnr_metrics_from_grad_vectors([g0, g1])
        self.assertTrue(math.isfinite(m["gsnr_encoder"]))
        self.assertLess(m["gsnr_encoder"], 1.0)

    def test_gsnr_nan_when_k_lt_2(self) -> None:
        m = gsnr_metrics_from_grad_vectors([torch.ones(3)])
        self.assertTrue(math.isnan(m["gsnr_encoder"]))


if __name__ == "__main__":
    unittest.main()
