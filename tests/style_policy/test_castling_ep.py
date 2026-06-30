"""Tests for the castling/ep feature in BoardEncoder (use_castling_ep flag).

Three checks:
  (a) use_castling_ep=False: output is byte-identical to a same-shaped encoder without the flag.
  (b) use_castling_ep=True: two board tensors differing ONLY in castling (planes 13-16) and/or
      ep (plane 17) produce DIFFERENT (cls, squares) outputs.
  (c) At init (zero-init castle_emb, zero ep_emb) the True-encoder output equals the
      False-encoder output (the no-op-until-trained property).
"""
import torch
import pytest
from style_policy.board_encoder import BoardEncoder


def _make_encoder(**kwargs) -> BoardEncoder:
    return BoardEncoder(d_model=16, n_layers=1, nhead=2, dim_feedforward=32, dropout=0.0, **kwargs)


def _rand_board(B: int = 2, seed: int = 42) -> torch.Tensor:
    """Return a random (B,8,8,18) board tensor with planes 13-17 zeroed."""
    g = torch.Generator()
    g.manual_seed(seed)
    t = torch.zeros(B, 8, 8, 18)
    # fill planes 0-12 with random floats (pieces + turn)
    t[..., :13] = torch.rand(B, 8, 8, 13, generator=g)
    # planes 13-17 stay zero (no castling, no ep)
    return t


# ---------------------------------------------------------------------------
# (a) use_castling_ep=False is byte-identical to the baseline encoder
# ---------------------------------------------------------------------------
def test_false_flag_same_as_no_flag():
    """With use_castling_ep=False the forward path must be identical to an encoder without the flag."""
    board = _rand_board()
    enc_base = _make_encoder()  # default False
    enc_false = _make_encoder(use_castling_ep=False)

    # Copy weights from enc_base to enc_false (both have identical param shapes at False)
    enc_false.load_state_dict(enc_base.state_dict())

    enc_base.eval()
    enc_false.eval()
    with torch.no_grad():
        cls_base, sq_base = enc_base(board)
        cls_false, sq_false = enc_false(board)

    assert torch.equal(cls_base, cls_false), "cls differs between default and use_castling_ep=False"
    assert torch.equal(sq_base, sq_false), "squares differ between default and use_castling_ep=False"


# ---------------------------------------------------------------------------
# (b) use_castling_ep=True: different castling/ep planes => different output
# ---------------------------------------------------------------------------
def test_true_flag_different_planes_produce_different_output():
    """With use_castling_ep=True, changing castling or ep planes changes the output."""
    enc = _make_encoder(use_castling_ep=True)

    # First, check that castle_emb has at least some non-zero weights after random init
    # (the zero-init only affects weight, but let's use non-zero weights to ensure sensitivity)
    # Give castle_emb non-trivial weights so a change in mask is detectable
    with torch.no_grad():
        torch.nn.init.normal_(enc.castle_emb.weight, mean=0.0, std=0.1)
        enc.ep_emb.data.fill_(0.1)

    board_a = _rand_board(B=1)
    board_b = board_a.clone()

    # board_b has castling rights on plane 13 set (broadcast over all squares)
    board_b[0, :, :, 13] = 1.0
    # and an en-passant square on plane 17
    board_b[0, 4, 4, 17] = 1.0

    enc.eval()
    with torch.no_grad():
        cls_a, sq_a = enc(board_a)
        cls_b, sq_b = enc(board_b)

    assert not torch.equal(cls_a, cls_b), "cls should differ when castling/ep planes differ"
    assert not torch.equal(sq_a, sq_b), "squares should differ when ep planes differ"


# ---------------------------------------------------------------------------
# (c) At init the True-encoder output equals the False-encoder output (zero-init no-op)
# ---------------------------------------------------------------------------
def test_true_flag_at_init_is_noop():
    """At init (zero castle_emb weights, zero ep_emb) the True-encoder output must equal
    the False-encoder output given the same base weights."""
    enc_false = _make_encoder(use_castling_ep=False)
    enc_true = _make_encoder(use_castling_ep=True)

    # Copy the shared weights from enc_false into enc_true
    # (castle_emb and ep_emb exist only in enc_true; they are zero-initialized)
    sd_false = enc_false.state_dict()
    sd_true = enc_true.state_dict()
    for k in sd_false:
        sd_true[k] = sd_false[k].clone()
    enc_true.load_state_dict(sd_true)

    # Use a board with non-zero castling and ep planes to exercise the branch
    board = _rand_board(B=2)
    board[:, :, :, 13] = 1.0   # some castling rights
    board[:, :, :, 15] = 1.0
    board[:, 3, 3, 17] = 1.0   # ep square

    enc_false.eval()
    enc_true.eval()
    with torch.no_grad():
        cls_f, sq_f = enc_false(board)
        cls_t, sq_t = enc_true(board)

    assert torch.allclose(cls_f, cls_t, atol=1e-6), \
        f"cls should be equal at init (zero-init no-op); max diff={( cls_f - cls_t).abs().max()}"
    assert torch.allclose(sq_f, sq_t, atol=1e-6), \
        f"squares should be equal at init (zero-init no-op); max diff={(sq_f - sq_t).abs().max()}"
