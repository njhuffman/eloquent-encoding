"""Tests for last-move history markers in BoardEncoder (use_last_move flag).

Four checks:
  (a) use_last_move=False: output identical to building without the flag (hist ignored).
  (b) use_last_move=True + zero-init: output equals the no-hist case (no-op until trained).
  (c) After randomizing from_emb/to_emb/cap_emb, two hist inputs differing in ONE present
      ply give different (cls, squares).
  (d) An all-absent (-1) hist gives the same output as zero-init/no-hist (absent contributes
      nothing — guards the negative-index bug).
"""
import torch
import pytest
from style_policy.board_encoder import BoardEncoder


def _make_encoder(**kwargs) -> BoardEncoder:
    return BoardEncoder(d_model=16, n_layers=1, nhead=2, dim_feedforward=32, dropout=0.0, **kwargs)


def _rand_board(B: int = 2, seed: int = 42) -> torch.Tensor:
    """Return a random (B,8,8,18) board tensor."""
    g = torch.Generator()
    g.manual_seed(seed)
    t = torch.zeros(B, 8, 8, 18)
    t[..., :13] = torch.rand(B, 8, 8, 13, generator=g)
    return t


def _make_hist(B: int, n_ply: int = 4, seed: int = 7) -> tuple:
    """Return (hf, ht, hc) each (B, n_ply) int64 with realistic values (squares 0-63, caps 0-5)."""
    g = torch.Generator()
    g.manual_seed(seed)
    hf = torch.randint(0, 64, (B, n_ply), generator=g)
    ht = torch.randint(0, 64, (B, n_ply), generator=g)
    hc = torch.randint(0, 6, (B, n_ply), generator=g)
    return hf, ht, hc


def _all_absent_hist(B: int, n_ply: int = 4) -> tuple:
    """Return hist where all plies are absent (from=-1, to=-1, cap=0)."""
    hf = torch.full((B, n_ply), -1, dtype=torch.long)
    ht = torch.full((B, n_ply), -1, dtype=torch.long)
    hc = torch.zeros(B, n_ply, dtype=torch.long)
    return hf, ht, hc


# ---------------------------------------------------------------------------
# (a) use_last_move=False: output identical regardless of hist passed in
# ---------------------------------------------------------------------------
def test_false_flag_hist_ignored():
    """With use_last_move=False, output must be identical whether hist is passed or not."""
    board = _rand_board()
    enc = _make_encoder()  # default use_last_move=False
    hist = _make_hist(B=2)

    enc.eval()
    with torch.no_grad():
        cls_no, sq_no = enc(board, hist=None)
        cls_wh, sq_wh = enc(board, hist=hist)

    assert torch.equal(cls_no, cls_wh), "cls differs when hist passed to use_last_move=False encoder"
    assert torch.equal(sq_no, sq_wh), "squares differ when hist passed to use_last_move=False encoder"


# ---------------------------------------------------------------------------
# (b) use_last_move=True + zero-init: same output as False-encoder (no-op)
# ---------------------------------------------------------------------------
def test_true_flag_zero_init_is_noop():
    """At init (zero from_emb/to_emb/cap_emb) the True-encoder output must equal
    the False-encoder output (no-op until trained)."""
    enc_false = _make_encoder(use_last_move=False)
    enc_true = _make_encoder(use_last_move=True, n_history_ply=4)

    # Copy shared weights from enc_false into enc_true
    sd_false = enc_false.state_dict()
    sd_true = enc_true.state_dict()
    for k in sd_false:
        sd_true[k] = sd_false[k].clone()
    enc_true.load_state_dict(sd_true)

    board = _rand_board(B=2)
    hist = _make_hist(B=2)

    enc_false.eval()
    enc_true.eval()
    with torch.no_grad():
        cls_f, sq_f = enc_false(board)
        cls_t, sq_t = enc_true(board, hist=hist)

    assert torch.allclose(cls_f, cls_t, atol=1e-6), \
        f"cls should be equal at init (zero-init no-op); max diff={(cls_f - cls_t).abs().max()}"
    assert torch.allclose(sq_f, sq_t, atol=1e-6), \
        f"squares should be equal at init (zero-init no-op); max diff={(sq_f - sq_t).abs().max()}"


# ---------------------------------------------------------------------------
# (c) Randomized weights: different present ply => different output
# ---------------------------------------------------------------------------
def test_different_hist_gives_different_output():
    """After randomizing from_emb/to_emb/cap_emb, two hists differing in ONE ply
    must produce different (cls, squares)."""
    enc = _make_encoder(use_last_move=True, n_history_ply=4)

    # Randomize the history-specific parameters
    with torch.no_grad():
        torch.nn.init.normal_(enc.from_emb, std=0.1)
        torch.nn.init.normal_(enc.to_emb, std=0.1)
        torch.nn.init.normal_(enc.cap_emb.weight, std=0.1)

    board = _rand_board(B=2)
    hf_a, ht_a, hc_a = _make_hist(B=2, seed=7)
    hf_b, ht_b, hc_b = hf_a.clone(), ht_a.clone(), hc_a.clone()
    # Change ply 0 for both samples — different from/to/cap squares
    hf_b[:, 0] = (hf_a[:, 0] + 10) % 64
    ht_b[:, 0] = (ht_a[:, 0] + 10) % 64
    hc_b[:, 0] = (hc_a[:, 0] + 1) % 6

    enc.eval()
    with torch.no_grad():
        cls_a, sq_a = enc(board, hist=(hf_a, ht_a, hc_a))
        cls_b, sq_b = enc(board, hist=(hf_b, ht_b, hc_b))

    # At least one of cls or squares should differ
    output_a = torch.cat([cls_a.flatten(), sq_a.flatten()])
    output_b = torch.cat([cls_b.flatten(), sq_b.flatten()])
    assert not torch.equal(output_a, output_b), \
        "Output should differ when ply-0 of hist is changed (non-zero randomized embeddings)"


# ---------------------------------------------------------------------------
# (d) All-absent hist contributes nothing (guards the -1 negative-index bug)
# ---------------------------------------------------------------------------
def test_all_absent_hist_is_noop():
    """All-absent hist (-1 for all from/to squares) must produce the same output as
    zero-init / no-hist baseline. This guards the negative-index bug: -1 must never
    index into tok."""
    enc_ref = _make_encoder(use_last_move=False)
    enc_hist = _make_encoder(use_last_move=True, n_history_ply=4)

    # Give enc_hist non-trivial history embeddings so any absent-ply leak would be visible
    with torch.no_grad():
        torch.nn.init.normal_(enc_hist.from_emb, std=0.5)
        torch.nn.init.normal_(enc_hist.to_emb, std=0.5)
        torch.nn.init.normal_(enc_hist.cap_emb.weight, std=0.5)

    # Copy shared weights
    sd_ref = enc_ref.state_dict()
    sd_hist = enc_hist.state_dict()
    for k in sd_ref:
        sd_hist[k] = sd_ref[k].clone()
    enc_hist.load_state_dict(sd_hist)

    board = _rand_board(B=3)
    absent = _all_absent_hist(B=3)

    enc_ref.eval()
    enc_hist.eval()
    with torch.no_grad():
        cls_ref, sq_ref = enc_ref(board)
        cls_hist, sq_hist = enc_hist(board, hist=absent)

    assert torch.allclose(cls_ref, cls_hist, atol=1e-6), \
        f"cls differs for all-absent hist; max diff={(cls_ref - cls_hist).abs().max()}"
    assert torch.allclose(sq_ref, sq_hist, atol=1e-6), \
        f"squares differ for all-absent hist; max diff={(sq_ref - sq_hist).abs().max()}"


# ---------------------------------------------------------------------------
# (e) Partial absence: mixed present/absent plies per sample
# ---------------------------------------------------------------------------
def test_partial_absent_per_sample():
    """Rows with some plies absent and some present: absent plies must not bleed."""
    enc = _make_encoder(use_last_move=True, n_history_ply=4)
    with torch.no_grad():
        torch.nn.init.normal_(enc.from_emb, std=0.1)
        torch.nn.init.normal_(enc.to_emb, std=0.1)
        torch.nn.init.normal_(enc.cap_emb.weight, std=0.1)

    board = _rand_board(B=2)

    # hist_a: ply 0 present, plies 1-3 absent for row 0; all absent for row 1
    hf = torch.tensor([[10, -1, -1, -1], [-1, -1, -1, -1]], dtype=torch.long)
    ht = torch.tensor([[20, -1, -1, -1], [-1, -1, -1, -1]], dtype=torch.long)
    hc = torch.tensor([[2,   0,  0,  0], [ 0,  0,  0,  0]], dtype=torch.long)

    # hist_b: same but ply 0 for row 0 is DIFFERENT
    hf2 = hf.clone()
    hf2[0, 0] = 30
    ht2 = ht.clone()
    ht2[0, 0] = 40

    enc.eval()
    with torch.no_grad():
        cls_a, sq_a = enc(board, hist=(hf, ht, hc))
        cls_b, sq_b = enc(board, hist=(hf2, ht2, hc))

    # Row 1 (all absent) should be identical across both hist inputs
    assert torch.allclose(cls_a[1], cls_b[1], atol=1e-6), \
        "Row 1 (all-absent) should not differ between hist_a and hist_b"
    assert torch.allclose(sq_a[1], sq_b[1], atol=1e-6), \
        "Row 1 squares (all-absent) should not differ between hist_a and hist_b"

    # Row 0 should differ (different ply-0)
    assert not torch.equal(sq_a[0], sq_b[0]), \
        "Row 0 should differ when ply-0 from/to squares change"
