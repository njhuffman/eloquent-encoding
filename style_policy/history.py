"""History-horizon dropout for last-move history tensors.

During training, randomly truncate each sample's ply history to a shorter
newest-first prefix, simulating inference with a shorter context window.

Absent-ply sentinel: from=-1, to=-1, cap=0.
History is prefix-packed newest-first: present plies occupy indices 0..K-1;
indices K..3 are absent.
"""
from __future__ import annotations

import torch


def horizon_dropout(
    hist_from: torch.Tensor,
    hist_to: torch.Tensor,
    hist_cap: torch.Tensor,
    p: float,
    gen: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply history-horizon dropout in-place and return the (possibly modified) tensors.

    Per row, with probability *p*:
      - Compute available = number of present plies (hist_from[row] >= 0).
      - If available == 0: leave unchanged (nothing to drop).
      - Otherwise draw K ~ Uniform{0 .. available-1} and set plies at indices
        >= K to absent (from=-1, to=-1, cap=0).  This keeps the newest K plies.

    With p=0 the function is a no-op (identity).
    The dropout mask is sampled with *gen* (for reproducibility in tests).

    Args:
        hist_from: (B, n_ply) int64 — from-square or -1 when absent.
        hist_to:   (B, n_ply) int64 — to-square or -1 when absent.
        hist_cap:  (B, n_ply) int64 — captured piece type 0-5 (0=none).
        p:         probability of applying dropout to a given row.
        gen:       optional torch.Generator for reproducible sampling.

    Returns:
        (hf, ht, hc) — same tensors, potentially modified in-place.
    """
    if p <= 0.0:
        return hist_from, hist_to, hist_cap

    B, n_ply = hist_from.shape

    # available[i] = number of present plies in row i
    available = (hist_from >= 0).sum(dim=1)  # (B,)

    # Sample Bernoulli mask: which rows get dropout applied?
    if p >= 1.0:
        apply_mask = torch.ones(B, dtype=torch.bool, device=hist_from.device)
    else:
        apply_mask = torch.bernoulli(
            torch.full((B,), p, dtype=torch.float32, device=hist_from.device),
            generator=gen,
        ).bool()

    # For rows where available=0, dropout is a no-op regardless
    apply_mask = apply_mask & (available > 0)

    rows_to_process = apply_mask.nonzero(as_tuple=False).squeeze(1)  # (M,)
    if rows_to_process.numel() == 0:
        return hist_from, hist_to, hist_cap

    # For each selected row, draw K ~ Uniform{0 .. available[row]-1}
    avail_selected = available[rows_to_process].float()  # (M,)
    noise = torch.rand(rows_to_process.shape[0], generator=gen, device=hist_from.device)
    # K = floor(noise * available), so K in {0 .. available-1}
    K = (noise * avail_selected).long()  # (M,)

    # For each ply index, zero out plies whose index >= K
    ply_idx = torch.arange(n_ply, device=hist_from.device).unsqueeze(0)  # (1, n_ply)
    K_expanded = K.unsqueeze(1)  # (M, 1)
    drop = ply_idx >= K_expanded  # (M, n_ply) True = should be absent

    row_idx = rows_to_process  # (M,)
    # Flatten indices for scatter: row × n_ply positions to overwrite
    # Use a loop over rows (M is at most B, typically small enough; also avoids complex indexing)
    # Vectorised approach: build a full (B, n_ply) drop mask
    full_drop = torch.zeros(B, n_ply, dtype=torch.bool, device=hist_from.device)
    full_drop[row_idx] = drop

    hist_from[full_drop] = -1
    hist_to[full_drop] = -1
    hist_cap[full_drop] = 0

    return hist_from, hist_to, hist_cap


def binary_history_dropout(
    hist_from: torch.Tensor,
    hist_to: torch.Tensor,
    hist_cap: torch.Tensor,
    p: float,
    gen: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Binary 'full history or none' dropout: per row, with probability *p* drop ALL plies
    (set every ply absent: from=-1, to=-1, cap=0); otherwise leave the row unchanged.

    Unlike :func:`horizon_dropout` there are no intermediate horizons — a row is either kept
    in full or blanked entirely. This matches how history is used at inference (full context
    when available, none for puzzles/FEN-only) without training robustness to arbitrary K.

    With p=0 the function is a no-op (identity). Modifies in-place and returns the tensors.
    The dropout mask is sampled with *gen* (for reproducibility in tests).
    """
    if p <= 0.0:
        return hist_from, hist_to, hist_cap

    B = hist_from.shape[0]
    if p >= 1.0:
        drop_rows = torch.ones(B, dtype=torch.bool, device=hist_from.device)
    else:
        drop_rows = torch.bernoulli(
            torch.full((B,), p, dtype=torch.float32, device=hist_from.device),
            generator=gen,
        ).bool()

    hist_from[drop_rows] = -1
    hist_to[drop_rows] = -1
    hist_cap[drop_rows] = 0
    return hist_from, hist_to, hist_cap
