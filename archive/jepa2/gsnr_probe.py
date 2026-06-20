"""Encoder-only gradient signal-to-noise ratio (GSNR) from repeated minibatch grads."""

from __future__ import annotations

import torch


def flatten_encoder_online_grads(model: torch.nn.Module) -> torch.Tensor:
    """Concatenate ``encoder_online`` parameter gradients into one 1-D float32 vector."""
    if not hasattr(model, "encoder_online"):
        raise AttributeError("model has no encoder_online; GSNR probe unsupported")
    parts: list[torch.Tensor] = []
    for p in model.encoder_online.parameters():
        if p.grad is None:
            continue
        parts.append(p.grad.detach().flatten().float())
    if not parts:
        return torch.zeros(1, dtype=torch.float32)
    return torch.cat(parts, dim=0)


def gsnr_metrics_from_grad_vectors(
    grads: list[torch.Tensor],
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    """
    ``grads``: K vectors (same length), e.g. CPU float32.

    Signal ``S = ||mean||^2``, noise ``N = mean_i ||g_i - mean||^2``, ``gsnr_encoder = S / (N + eps)``.
    """
    if len(grads) < 2:
        return {
            "gsnr_encoder": float("nan"),
            "gsnr_signal": float("nan"),
            "gsnr_noise": float("nan"),
            "gsnr_grad_norm_mean": float("nan"),
        }
    stack = torch.stack([g.float() for g in grads], dim=0)
    g_mean = stack.mean(dim=0)
    s = float(g_mean.pow(2).sum().item())
    n = float((stack - g_mean).pow(2).sum(dim=1).mean().item())
    norms = torch.linalg.vector_norm(stack, dim=1)
    gsnr = s / (n + eps) if n == n else float("nan")
    return {
        "gsnr_encoder": float(gsnr) if gsnr == gsnr else float("nan"),
        "gsnr_signal": s,
        "gsnr_noise": n,
        "gsnr_grad_norm_mean": float(norms.mean().item()),
    }
