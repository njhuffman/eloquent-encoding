import numpy as np, chess
from style_policy.board_encode import legal_move_matrix, legal_from_u64
import torch
from style_policy.pointer_head import PointerHead, joint_ce

def test_legal_move_matrix_matches_board():
    board = chess.Board()
    m = legal_move_matrix(board)
    pairs = {(mv.from_square, mv.to_square) for mv in board.legal_moves}
    assert m.sum() == len(pairs)
    for (fr, to) in pairs:
        assert m[fr, to]
    froms = {i for i in range(64) if m[i].any()}
    lf = legal_from_u64(board)
    assert froms == {i for i in range(64) if (lf >> i) & 1}

def test_pointer_shapes_and_cls():
    h = PointerHead(d_model=32, d_head=16, n_heads=2, use_cls=True)
    sq = torch.randn(4, 64, 32); cls = torch.randn(4, 32)
    assert h(sq, cls).shape == (4, 64, 64)
    try:
        h(sq, None); assert False
    except ValueError:
        pass

def test_pointer_no_cls_ignores_cls():
    h = PointerHead(d_model=32, use_cls=False)
    sq = torch.randn(2, 64, 32)
    assert torch.allclose(h(sq, None), h(sq, torch.randn(2, 32)))

def test_joint_ce_masks_and_runs():
    h = PointerHead(d_model=32, use_cls=False)
    sq = torch.randn(2, 64, 32)
    logits = h(sq, None)
    legal = torch.zeros(2, 64, 64, dtype=torch.bool)
    legal[0, 8, 16] = True; legal[0, 8, 24] = True; legal[1, 1, 18] = True
    fr = torch.tensor([8, 1]); to = torch.tensor([16, 18])
    loss = joint_ce(logits, fr, to, legal)
    assert torch.isfinite(loss)
    flat = logits.masked_fill(~legal, float("-inf")).view(2, -1)
    idx = flat.argmax(-1)
    assert bool(legal[0, idx[0] // 64, idx[0] % 64]) and bool(legal[1, idx[1] // 64, idx[1] % 64])
