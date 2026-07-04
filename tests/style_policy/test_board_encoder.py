import torch
import h5py
from style_policy.board_encoder import BoardEncoder
from style_policy.packed_codec import packed_to_board_tensor


def _boards(n=4):
    with h5py.File("/mnt/eloquence_bulk/databases/j3_training_1M.h5", "r") as f:
        packed = torch.from_numpy(f["packed_pre"][0:n].astype("uint8"))
    return packed_to_board_tensor(packed)


def test_encoder_output_shapes():
    enc = BoardEncoder(d_model=64, n_layers=2, nhead=4, dim_feedforward=128, dropout=0.0)
    cls, squares = enc(_boards(4))
    assert cls.shape == (4, 64)
    assert squares.shape == (4, 64, 64)


def test_turn_changes_encoding():
    # Boards with plane 12 = 0.0 (black to move) vs 1.0 (white to move) must produce
    # different CLS outputs, confirming the turn token is live in the encoder.
    enc = BoardEncoder(d_model=64, n_layers=2, nhead=4, dim_feedforward=128, dropout=0.0).eval()
    base = _boards(2)
    a = base.clone()
    b = base.clone()
    a[..., 12] = 0.0  # black to move
    b[..., 12] = 1.0  # white to move
    with torch.no_grad():
        cls_a, _ = enc(a)
        cls_b, _ = enc(b)
    assert not torch.allclose(cls_a, cls_b), "CLS should differ when side-to-move changes"


def test_gab_eval_matches_train():
    # Regression: with GAB, PyTorch's fused MHA fast path silently mishandles the batched
    # float attention mask in eval() while train() takes the slow path -> the two modes
    # diverged and inference (K-sweep, bot) got garbage. With dropout=0 the modes must be
    # numerically identical; board_encoder forces the slow path when use_gab to guarantee it.
    enc = BoardEncoder(d_model=64, n_layers=2, nhead=4, dim_feedforward=128, dropout=0.0,
                       use_gab=True).eval()
    # non-trivial GAB weights so the mask actually bites (proj3 is zero-init = no-op otherwise)
    with torch.no_grad():
        enc.gab_proj3.weight.normal_(0, 0.02)
        enc.gab_proj3.bias.normal_(0, 0.02)
    boards = _boards(4)
    with torch.no_grad():
        enc.eval();  cls_eval, sq_eval = enc(boards)
        enc.train(); cls_train, sq_train = enc(boards)
    assert torch.allclose(cls_eval, cls_train, atol=1e-5), "GAB eval() diverged from train()"
    assert torch.allclose(sq_eval, sq_train, atol=1e-5), "GAB eval() diverged from train()"
