"""Tests for jepa2.sam (Sharpness-Aware Minimization helpers)."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from jepa2.sam import (
    global_grad_l2_norm,
    sam_apply_perturbation,
    sam_build_perturbations,
    sam_revert_perturbation,
)


class Toy2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor([3.0, 4.0]))
        self.b = nn.Parameter(torch.tensor([1.0]))


class SamTests(unittest.TestCase):
    def test_global_norm_and_epsilon_direction(self) -> None:
        m = Toy2()
        m.w.grad = torch.tensor([3.0, 4.0])
        m.b.grad = torch.tensor([0.0])
        norm = global_grad_l2_norm([m.w, m.b])
        self.assertAlmostEqual(float(norm.item()), 5.0, places=6)

        rho = 0.2
        pert = sam_build_perturbations([m.w, m.b], rho)
        self.assertEqual(len(pert), 2)
        eps_w = next(e for p, e in pert if p is m.w)
        eps_b = next(e for p, e in pert if p is m.b)
        g_w = m.w.grad.detach()
        self.assertTrue(torch.allclose(eps_w, rho * g_w / 5.0))
        self.assertTrue(torch.allclose(eps_b, torch.zeros_like(eps_b)))

    def test_apply_revert_restores_weights(self) -> None:
        m = Toy2()
        m.w.grad = torch.tensor([5.0, 0.0])
        m.b.grad = torch.tensor([0.0])
        rho = 0.1
        pert = sam_build_perturbations([m.w, m.b], rho)
        w0 = m.w.data.clone()
        b0 = m.b.data.clone()
        sam_apply_perturbation(pert)
        self.assertFalse(torch.equal(m.w.data, w0))
        sam_revert_perturbation(pert)
        self.assertTrue(torch.allclose(m.w.data, w0))
        self.assertTrue(torch.allclose(m.b.data, b0))

    def test_rho_zero_or_bad_norm_returns_empty(self) -> None:
        m = Toy2()
        m.w.grad = torch.tensor([1.0, 0.0])
        m.b.grad = torch.tensor([0.0])
        self.assertEqual(sam_build_perturbations([m.w, m.b], 0.0), [])
        m.w.grad = torch.zeros(2)
        m.b.grad = torch.zeros(1)
        self.assertEqual(sam_build_perturbations([m.w, m.b], 1.0), [])

    def test_global_norm_skips_none_grad(self) -> None:
        m = Toy2()
        m.w.grad = torch.tensor([1.0, 0.0])
        m.b.grad = None
        n = float(global_grad_l2_norm([m.w, m.b]).item())
        self.assertAlmostEqual(n, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
