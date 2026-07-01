"""Tests for style_policy/history.py: binary_history_dropout.

Binary "full history or none": per row, with prob p drop ALL plies (K=0), else keep the row
unchanged. No intermediate horizons (contrast horizon_dropout). Checks:
  (1) p=0.0 is identity.
  (2) p=1.0 drops every row fully (all plies absent -1/-1/0).
  (3) 0<p<1: each row is EITHER unchanged OR fully absent — never partially truncated.
  (4) already-absent rows stay absent under any p.
  (5) seeded determinism.
"""
import torch
from style_policy.history import binary_history_dropout


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _make_hist_full(B: int = 6):
    hf = torch.arange(B * 4).reshape(B, 4) % 64
    ht = (torch.arange(B * 4).reshape(B, 4) + 1) % 64
    hc = torch.ones(B, 4, dtype=torch.long)  # nonzero caps so "absent" (0) is distinguishable
    return hf, ht, hc


def test_p0_identity():
    hf, ht, hc = _make_hist_full()
    o = binary_history_dropout(hf.clone(), ht.clone(), hc.clone(), p=0.0, gen=_gen(0))
    assert torch.equal(o[0], hf) and torch.equal(o[1], ht) and torch.equal(o[2], hc)


def test_p1_drops_all_rows_fully():
    B = 16
    hf, ht, hc = _make_hist_full(B)
    of, ot, oc = binary_history_dropout(hf.clone(), ht.clone(), hc.clone(), p=1.0, gen=_gen(1))
    assert (of == -1).all() and (ot == -1).all() and (oc == 0).all()


def test_binary_never_partial():
    """Each row is either fully unchanged or fully absent — no intermediate truncation."""
    B = 200
    hf, ht, hc = _make_hist_full(B)
    of, ot, oc = binary_history_dropout(hf.clone(), ht.clone(), hc.clone(), p=0.5, gen=_gen(7))
    dropped = 0
    for r in range(B):
        row_absent = bool((of[r] == -1).all())
        row_kept = torch.equal(of[r], hf[r]) and torch.equal(ot[r], ht[r]) and torch.equal(oc[r], hc[r])
        assert row_absent ^ row_kept, f"row {r} is neither fully-absent nor fully-kept (partial!)"
        dropped += int(row_absent)
    assert 0 < dropped < B, f"expected a mix at p=0.5, got {dropped}/{B} dropped"


def test_already_absent_rows_stay_absent():
    B = 8
    hf = torch.full((B, 4), -1, dtype=torch.long)
    ht = torch.full((B, 4), -1, dtype=torch.long)
    hc = torch.zeros(B, 4, dtype=torch.long)
    of, ot, oc = binary_history_dropout(hf.clone(), ht.clone(), hc.clone(), p=1.0, gen=_gen(5))
    assert torch.equal(of, hf) and torch.equal(ot, ht) and torch.equal(oc, hc)


def test_seeded_determinism():
    hf, ht, hc = _make_hist_full(10)
    a = binary_history_dropout(hf.clone(), ht.clone(), hc.clone(), p=0.5, gen=_gen(123))
    b = binary_history_dropout(hf.clone(), ht.clone(), hc.clone(), p=0.5, gen=_gen(123))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])
