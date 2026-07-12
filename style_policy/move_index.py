"""Flat move space for Pass-2 heads: the 1792 promotion-free (from,to) pairs.

Every legal chess move's (from,to) is a subset of {queen-reachable} ∪ {knight-reachable} on an empty
board (rook/bishop/king/pawn moves ⊂ queen; knight separate) -> 1792 distinct pairs. Promotions are
ignored: a pawn reaching the back rank is just its (from,to) queen-pair, interpreted as a queen
promotion at inference. The move heads emit one 1792-way softmax (flat, not factored).

Exports:
  N_MOVES = 1792
  IDX_FROM, IDX_TO : int64[1792]           from/to square of each flat index (for gather from a 64x64 score matrix)
  move_to_index(from_sq, to_sq) -> int     target encoding (drops promotion)
  legal_index_mask(board) -> bool[1792]    exact per-position legal mask (train-time, from a reconstructed board)
"""
from __future__ import annotations
import numpy as np
import chess


def _build_moves():
    seen = set()
    for f in chess.SQUARES:
        for piece in (chess.QUEEN, chess.KNIGHT):
            b = chess.Board(None)
            b.set_piece_at(f, chess.Piece(piece, chess.WHITE))
            for t in b.attacks(f):
                seen.add((f, t))
    return sorted(seen)                                    # deterministic order -> stable indices


MOVES = _build_moves()                                     # list[(from,to)], len 1792
N_MOVES = len(MOVES)
_MOVE_TO_IDX = {m: i for i, m in enumerate(MOVES)}
IDX_FROM = np.array([m[0] for m in MOVES], dtype=np.int64)
IDX_TO = np.array([m[1] for m in MOVES], dtype=np.int64)
IDX_FT = IDX_FROM * 64 + IDX_TO                             # flat index into a (64*64) score matrix
# Dense (from*64+to) -> flat index lookup (-1 if geometrically impossible), for vectorized target encoding.
_FT_TO_IDX = np.full(64 * 64, -1, dtype=np.int64)
for _i, (_f, _t) in enumerate(MOVES):
    _FT_TO_IDX[_f * 64 + _t] = _i


def move_to_index(from_sq: int, to_sq: int) -> int:
    return _MOVE_TO_IDX[(int(from_sq), int(to_sq))]


def move_to_index_arr(from_sq, to_sq):
    """Vectorized: int arrays of from/to squares -> int64 flat indices (-1 where impossible)."""
    return _FT_TO_IDX[np.asarray(from_sq, dtype=np.int64) * 64 + np.asarray(to_sq, dtype=np.int64)]


def legal_index_mask(board: chess.Board) -> np.ndarray:
    m = np.zeros(N_MOVES, dtype=bool)
    for mv in board.legal_moves:
        m[_MOVE_TO_IDX[(mv.from_square, mv.to_square)]] = True
    return m


if __name__ == "__main__":
    import time
    from style_policy.board_encode import packed_to_board
    import h5py
    print(f"N_MOVES = {N_MOVES}  (expect 1792)")
    print(f"IDX_FROM/IDX_TO shape {IDX_FROM.shape}, unique pairs {len({(int(f),int(t)) for f,t in zip(IDX_FROM,IDX_TO)})}")
    # every legal move of the start position maps into the table
    b = chess.Board()
    assert all((mv.from_square, mv.to_square) in _MOVE_TO_IDX for mv in b.legal_moves)
    mask = legal_index_mask(b); print(f"startpos legal-mask sum = {mask.sum()} (20 moves, 1 pawn-pair dup? -> <=20)")
    # promotion pair is present
    assert (chess.E7, chess.E8) in _MOVE_TO_IDX, "back-rank pawn pair missing"
    # throughput of reconstruct + mask (the dataloader cost that must beat ~1700 samp/s)
    f = h5py.File("/mnt/eloquence_bulk/databases/wdl_history_128M.h5", "r")
    pk = f["packed_pre"][:4000]
    t0 = time.time()
    for i in range(len(pk)):
        legal_index_mask(packed_to_board(pk[i].astype(np.uint8)))
    dt = time.time() - t0
    print(f"reconstruct+mask: {len(pk)/dt:.0f}/s single-thread  ({dt/len(pk)*1e6:.0f} us/pos)  [need >1700/s]")
