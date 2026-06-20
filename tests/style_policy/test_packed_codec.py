import chess
import numpy as np
import torch
from style_policy.packed_codec import board_tensor_to_packed, packed_to_board_tensor, PACKED_BOARD_LEN


def _board_tensor_from_fen(fen: str) -> np.ndarray:
    """Build an (8, 8, 18) float32 board tensor from a FEN string.

    Matches the channel layout documented in style_policy.packed_codec:
      Planes 0-11: piece occupancy (WP=0, WN=1, WB=2, WR=3, WQ=4, WK=5,
                                     BP=6, BN=7, BB=8, BR=9, BQ=10, BK=11)
      Plane 12: side-to-move (1.0 = White)
      Planes 13-16: castling rights (WK=13, WQ=14, BK=15, BQ=16)
      Plane 17: en-passant target square
    """
    board = chess.Board(fen)
    tensor = np.zeros((8, 8, 18), dtype=np.float32)

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        rank = sq // 8  # chess.A1=0 -> rank 0, chess.A8=56 -> rank 7
        file = sq % 8
        # plane index: piece_type-1 for white, piece_type-1+6 for black
        plane = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        tensor[rank, file, plane] = 1.0

    # Side to move: plane 12 is 1.0 everywhere if White's turn
    if board.turn == chess.WHITE:
        tensor[:, :, 12] = 1.0

    # Castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[:, :, 13] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[:, :, 14] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[:, :, 15] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[:, :, 16] = 1.0

    # En-passant target square
    if board.ep_square is not None:
        ep_rank = board.ep_square // 8
        ep_file = board.ep_square % 8
        tensor[ep_rank, ep_file, 17] = 1.0

    return tensor


def test_packed_len_constant():
    assert PACKED_BOARD_LEN == 34


def test_packed_roundtrip_startpos():
    """Pack a start-position board tensor and unpack it; the result must equal the original."""
    original = _board_tensor_from_fen(chess.STARTING_FEN)
    packed = board_tensor_to_packed(original)
    assert packed.shape == (PACKED_BOARD_LEN,)

    # packed_to_board_tensor accepts a batched tensor, returns (N, 8, 8, 18)
    restored_batch = packed_to_board_tensor(torch.from_numpy(packed[np.newaxis, :]))
    assert restored_batch.shape == (1, 8, 8, 18)
    restored = restored_batch[0].numpy()

    np.testing.assert_array_equal(
        restored, original,
        err_msg="pack→unpack round-trip failed for start position"
    )


def test_packed_roundtrip_midgame():
    """Round-trip a mid-game position with EP square and partial castling rights."""
    # 1.e4 e5 — after 1.e4 White to move, EP square at e6
    fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"
    original = _board_tensor_from_fen(fen)
    packed = board_tensor_to_packed(original)
    restored_batch = packed_to_board_tensor(torch.from_numpy(packed[np.newaxis, :]))
    restored = restored_batch[0].numpy()
    np.testing.assert_array_equal(restored, original, err_msg="round-trip failed for e4 e5 position")


def test_decode_real_row_matches_legal_from():
    import h5py
    with h5py.File("/mnt/eloquence_bulk/databases/j3_training_1M.h5", "r") as f:
        packed = torch.from_numpy(f["packed_pre"][0:4].astype("uint8"))
        from_sq = f["from_sq"][0:4].astype("int64")
        from_legal = f["from_legal_u64"][0:4].astype("uint64")
    board = packed_to_board_tensor(packed)
    assert board.shape == (4, 8, 8, 18)
    # Ground-truth from_sq must be a legal origin (bit set in from_legal_u64).
    for i in range(4):
        assert (int(from_legal[i]) >> int(from_sq[i])) & 1 == 1
