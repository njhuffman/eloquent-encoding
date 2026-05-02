"""Round-trip tests for jepa3 packed board codec vs board_to_tensor / legal masks."""

from __future__ import annotations

import chess
import numpy as np
import pytest

from embedding.board_encoding import board_to_tensor
from jepa.move_row_codec import tensor_after_move
from jepa3.board_masks import legal_from_square_mask, legal_to_square_mask
from jepa3.packed_board_codec import (
    PACKED_BOARD_LEN,
    board_tensor_to_packed,
    legal_mask_float_to_u64,
    legal_masks_to_u64,
    packed_to_board_tensor,
    u64_pair_to_masks,
    u64_to_legal_mask_float,
)
from jepa3.packed_dataset import PackedMoveRowDataset
from jepa3.packed_h5 import PackedMoveH5Writer


def _assert_tensor_close(a: np.ndarray, b: np.ndarray) -> None:
    assert a.shape == b.shape
    np.testing.assert_allclose(a, b, atol=1e-6, rtol=0)


@pytest.mark.parametrize(
    "fen",
    [
        chess.STARTING_FEN,
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "8/8/8/3p4/2pP4/8/8/8 b - d3 0 1",
        "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    ],
)
def test_board_pack_roundtrip(fen: str) -> None:
    board = chess.Board(fen)
    t = board_to_tensor(board)
    p = board_tensor_to_packed(t)
    assert p.shape == (PACKED_BOARD_LEN,)
    t2 = packed_to_board_tensor(p)
    _assert_tensor_close(t, t2)


def test_legal_mask_u64_roundtrip() -> None:
    board = chess.Board(chess.STARTING_FEN)
    fm = legal_from_square_mask(board)
    tm = legal_to_square_mask(board, chess.E2)
    u0, u1 = legal_masks_to_u64(fm, tm)
    f2, t2 = u64_pair_to_masks(u0, u1)
    np.testing.assert_array_equal(fm, f2)
    np.testing.assert_array_equal(tm, t2)


def test_u64_single_roundtrip() -> None:
    m = np.zeros(64, dtype=np.float32)
    m[0] = 1.0
    m[63] = 1.0
    u = legal_mask_float_to_u64(m)
    m2 = u64_to_legal_mask_float(u)
    np.testing.assert_array_equal(m, m2)


def test_packed_h5_single_row_roundtrip(tmp_path) -> None:
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    t_pre = board_to_tensor(board)
    t_post = tensor_after_move(board, move)
    fm = legal_from_square_mask(board)
    tm = legal_to_square_mask(board, chess.E2)
    fu, tu = legal_masks_to_u64(fm, tm)
    out = tmp_path / "one.h5"
    with PackedMoveH5Writer(out) as w:
        w.append_row(
            packed_pre=board_tensor_to_packed(t_pre),
            packed_post=board_tensor_to_packed(t_post),
            from_legal_u64=fu,
            to_legal_u64=tu,
            from_sq=chess.E2,
            to_sq=chess.E4,
            promotion=0,
            elo_to_move=1500,
        )
    ds = PackedMoveRowDataset(out, np.array([0], dtype=np.int64))
    bt, bp, fmf, tmf, elo, fs, ts, pr = ds[0]
    np.testing.assert_allclose(bt, t_pre, atol=1e-6)
    np.testing.assert_allclose(bp, t_post, atol=1e-6)
    np.testing.assert_array_equal(fmf, fm)
    np.testing.assert_array_equal(tmf, tm)
    assert elo == 1500.0
    assert fs == chess.E2 and ts == chess.E4 and pr == 0
