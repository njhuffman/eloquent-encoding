"""Tests for style_policy/history.py: horizon_dropout.

Checks:
  (1) p=0.0 is identity — no rows changed.
  (2) p=1.0 truncates every row to a contiguous newest-prefix (no gaps; dropped plies become -1/-1/0).
  (3) K=0 is reachable (full row set to absent).
  (4) A row with fewer present plies (partial prefix) still yields a valid contiguous prefix.
  (5) Absent rows (available=0) are left as all-absent.
"""
import torch
import pytest
from style_policy.history import horizon_dropout


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _make_hist_full(B: int = 6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """All 4 plies present for every row."""
    hf = torch.arange(B * 4).reshape(B, 4) % 64  # distinct squares, all >= 0
    ht = (torch.arange(B * 4).reshape(B, 4) + 1) % 64
    hc = torch.zeros(B, 4, dtype=torch.long)
    return hf, ht, hc


def _available(hf: torch.Tensor) -> torch.Tensor:
    """Count present plies per row (hf >= 0)."""
    return (hf >= 0).sum(dim=1)


# ---------------------------------------------------------------------------
# (1) p=0.0 is identity
# ---------------------------------------------------------------------------
def test_p0_identity():
    hf, ht, hc = _make_hist_full(B=6)
    hf2, ht2, hc2 = horizon_dropout(hf.clone(), ht.clone(), hc.clone(), p=0.0, gen=_gen(0))
    assert torch.equal(hf, hf2), "hf changed under p=0"
    assert torch.equal(ht, ht2), "ht changed under p=0"
    assert torch.equal(hc, hc2), "hc changed under p=0"


# ---------------------------------------------------------------------------
# (2) p=1.0 always truncates; result is a contiguous newest-prefix (no gaps)
# ---------------------------------------------------------------------------
def test_p1_contiguous_prefix():
    """With p=1.0 and 4 available plies, every row is truncated to a strict sub-prefix
    (some K in 0..3).  The result must be prefix-valid: no present ply after an absent one."""
    B = 32
    hf, ht, hc = _make_hist_full(B=B)
    hf_out, ht_out, hc_out = horizon_dropout(
        hf.clone(), ht.clone(), hc.clone(), p=1.0, gen=_gen(42)
    )

    for row in range(B):
        present = hf_out[row] >= 0   # bool (4,)
        # prefix property: once a ply is absent, all later ones must be absent too
        saw_absent = False
        for i in range(4):
            if present[i].item():
                assert not saw_absent, (
                    f"row {row}: gap detected — ply {i} is present but an earlier ply was absent"
                )
            else:
                saw_absent = True
        # dropped plies are fully absent (from=-1, to=-1, cap=0)
        for i in range(4):
            if not present[i].item():
                assert hf_out[row, i].item() == -1, f"row {row} ply {i}: hf not -1"
                assert ht_out[row, i].item() == -1, f"row {row} ply {i}: ht not -1"
                assert hc_out[row, i].item() == 0,  f"row {row} ply {i}: hc not 0"


# ---------------------------------------------------------------------------
# (3) K=0 is reachable (full row becomes absent)
# ---------------------------------------------------------------------------
def test_k0_reachable():
    """With enough rows and p=1.0, at least one row should eventually get K=0 (all absent).
    available=4 so K ~ Uniform{0..3}: prob 1/4 per row, so 64 rows gives ~16 full-absent rows."""
    B = 64
    hf, ht, hc = _make_hist_full(B=B)
    hf_out, ht_out, hc_out = horizon_dropout(
        hf.clone(), ht.clone(), hc.clone(), p=1.0, gen=_gen(7)
    )
    all_absent_count = ((hf_out == -1).all(dim=1)).sum().item()
    assert all_absent_count >= 1, (
        "No row became fully absent under p=1.0; K=0 must be reachable "
        f"(got {all_absent_count} fully-absent rows from {B})"
    )


# ---------------------------------------------------------------------------
# (4) Partial-prefix rows (available < 4) still yield a valid prefix
# ---------------------------------------------------------------------------
def test_partial_prefix_stays_valid():
    """Rows with only 2 present plies (newest-first prefix) should still produce
    a valid prefix after dropout."""
    B = 20
    # Build hist where rows have plies 0,1 present and plies 2,3 absent
    hf = torch.stack([
        torch.tensor([10, 20, -1, -1], dtype=torch.long)
    ] * B)
    ht = torch.stack([
        torch.tensor([11, 21, -1, -1], dtype=torch.long)
    ] * B)
    hc = torch.stack([
        torch.tensor([1,  2,   0,  0], dtype=torch.long)
    ] * B)

    hf_out, ht_out, hc_out = horizon_dropout(
        hf.clone(), ht.clone(), hc.clone(), p=1.0, gen=_gen(99)
    )

    for row in range(B):
        saw_absent = False
        for i in range(4):
            if hf_out[row, i].item() >= 0:
                assert not saw_absent, f"row {row}: gap at ply {i}"
            else:
                saw_absent = True
                assert hf_out[row, i].item() == -1
                assert ht_out[row, i].item() == -1
                assert hc_out[row, i].item() == 0


# ---------------------------------------------------------------------------
# (5) All-absent rows (available=0) left unchanged
# ---------------------------------------------------------------------------
def test_all_absent_rows_unchanged():
    """Rows that are already fully absent (available=0) must not be altered."""
    B = 8
    hf = torch.full((B, 4), -1, dtype=torch.long)
    ht = torch.full((B, 4), -1, dtype=torch.long)
    hc = torch.zeros(B, 4, dtype=torch.long)

    hf_out, ht_out, hc_out = horizon_dropout(
        hf.clone(), ht.clone(), hc.clone(), p=1.0, gen=_gen(5)
    )
    assert torch.equal(hf, hf_out), "all-absent hf changed"
    assert torch.equal(ht, ht_out), "all-absent ht changed"
    assert torch.equal(hc, hc_out), "all-absent hc changed"


# ---------------------------------------------------------------------------
# (6) Seeded determinism: same gen seed → same output
# ---------------------------------------------------------------------------
def test_seeded_determinism():
    B = 10
    hf, ht, hc = _make_hist_full(B=B)
    out1 = horizon_dropout(hf.clone(), ht.clone(), hc.clone(), p=0.5, gen=_gen(123))
    out2 = horizon_dropout(hf.clone(), ht.clone(), hc.clone(), p=0.5, gen=_gen(123))
    assert torch.equal(out1[0], out2[0]) and torch.equal(out1[1], out2[1]) and torch.equal(out1[2], out2[2])
