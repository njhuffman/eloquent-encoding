"""Sharpness-Aware Minimization (SAM): global L2-norm perturbation on trainable parameters."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.nn import Parameter


def global_grad_l2_norm(params: Iterable[Parameter]) -> torch.Tensor:
    """Euclidean norm of the vector formed by concatenating all non-None ``.grad`` tensors."""
    sq = None
    for p in params:
        if p.grad is None:
            continue
        v = p.grad.detach().double().pow(2).sum()
        sq = v if sq is None else sq + v
    if sq is None:
        return torch.zeros((), dtype=torch.double)
    return torch.sqrt(sq)


def sam_build_perturbations(
    params: Iterable[Parameter],
    rho: float,
    *,
    eps_norm_floor: float = 1e-12,
) -> list[tuple[Parameter, torch.Tensor]]:
    """
    From current ``.grad`` tensors, build ``epsilon_p = rho * g_p / ||g||_2`` for each parameter.

    Under AMP gradient scaling, using scaled ``.grad`` gives the same ``epsilon`` as unscaled
    grads (direction and ``rho``-ball step in parameter space).

    Returns an empty list if ``rho <= 0`` or the global grad norm is below ``eps_norm_floor``
    (caller should fall back to a normal optimizer step).
    """
    if rho <= 0.0:
        return []
    plist = list(params)
    norm = global_grad_l2_norm(plist)
    n = float(norm.item())
    if not (n >= eps_norm_floor) or not (n == n):
        return []
    out: list[tuple[Parameter, torch.Tensor]] = []
    inv = rho / n
    for p in plist:
        if p.grad is None:
            continue
        g = p.grad.detach()
        eps = (g * inv).to(dtype=p.data.dtype, device=p.data.device)
        out.append((p, eps))
    return out


def sam_apply_perturbation(perturbations: list[tuple[Parameter, torch.Tensor]]) -> None:
    for p, eps in perturbations:
        p.data.add_(eps)


def sam_revert_perturbation(perturbations: list[tuple[Parameter, torch.Tensor]]) -> None:
    for p, eps in perturbations:
        p.data.sub_(eps)
