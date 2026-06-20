"""Pack / unpack (8, 8, 18) float32 board tensors matching the on-disk j3 HDF5 training format.

Board-tensor channel layout (C=18 planes, shape H×W×C with H=W=8):
  Planes 0–11  — piece occupancy (one-hot per square):
    0  white pawn     (WP)
    1  white knight   (WN)
    2  white bishop   (WB)
    3  white rook     (WR)
    4  white queen    (WQ)
    5  white king     (WK)
    6  black pawn     (BP)
    7  black knight   (BN)
    8  black bishop   (BB)
    9  black rook     (BR)
   10  black queen    (BQ)
   11  black king     (BK)
  Plane 12 — side to move: 1.0 everywhere if it is White's turn, 0.0 everywhere if Black's.
  Planes 13–16 — castling rights (broadcast over all squares):
   13  White kingside  (H1 rook eligible)
   14  White queenside (A1 rook eligible)
   15  Black kingside  (H8 rook eligible)
   16  Black queenside (A8 rook eligible)
  Plane 17 — en-passant target: 1.0 on the target square, 0.0 elsewhere.

Packed format (PACKED_BOARD_LEN = 34 bytes):
  Bytes 0–31  — 64 nibbles (4 bits each), two per byte, little-nibble first.
                Nibble value 0 = empty; 1–6 = WP..WK; 7–12 = BP..BK.
  Byte 32     — meta flags byte:
                  bit 0  side-to-move (1 = White)
                  bit 1  White kingside castling
                  bit 2  White queenside castling
                  bit 3  Black kingside castling
                  bit 4  Black queenside castling
  Byte 33     — en-passant square index (row*8+col, 255 = no EP).
"""

from __future__ import annotations

import numpy as np
import torch

# Board dimensions — inlined to keep this file self-contained (H=8, W=8, C=18).
BOARD_HEIGHT = 8
BOARD_WIDTH = 8
BOARD_CHANNELS = 18

# 32 bytes: 64 squares × 4 bits (piece id 0–12)
# 2 bytes: turn + castling flags + EP square index (255 = no EP)
PACKED_BOARD_LEN = 34
PACKED_LAYOUT_VERSION = 1

# Nibble encoding: 0 empty; 1–6 white P..K; 7–12 black P..K (matches plane index +1 split)


def board_tensor_to_packed(board: np.ndarray) -> np.ndarray:
    """
    ``board`` float32 (8, 8, 18) as from board_to_tensor.
    Returns uint8 vector of length PACKED_BOARD_LEN.
    """
    b = np.asarray(board, dtype=np.float32)
    if b.shape != (BOARD_HEIGHT, BOARD_WIDTH, BOARD_CHANNELS):
        raise ValueError(f"expected ({BOARD_HEIGHT},{BOARD_WIDTH},{BOARD_CHANNELS}), got {b.shape}")

    p12 = b[:, :, :12]
    amax = np.argmax(p12, axis=2)
    mx = np.max(p12, axis=2)
    empty = mx < 0.5
    nibbles2d = np.where(empty, 0, amax + 1).astype(np.uint8, copy=False)
    if np.any(nibbles2d > 15):
        raise ValueError("invalid nibble in board tensor > 15")
    flat = nibbles2d.ravel(order="C")

    out = np.zeros(PACKED_BOARD_LEN, dtype=np.uint8)
    for sq in range(64):
        nib = int(flat[sq])
        bi = sq // 2
        if sq % 2 == 0:
            out[bi] = np.uint8(nib & 0x0F)
        else:
            out[bi] |= np.uint8((nib & 0x0F) << 4)

    turn_w = float(b[0, 0, 12]) > 0.5
    meta = 0
    if turn_w:
        meta |= 1
    if float(np.max(b[:, :, 13])) > 0.5:
        meta |= 2
    if float(np.max(b[:, :, 14])) > 0.5:
        meta |= 4
    if float(np.max(b[:, :, 15])) > 0.5:
        meta |= 8
    if float(np.max(b[:, :, 16])) > 0.5:
        meta |= 16
    out[32] = np.uint8(meta)

    ep_sq = 255
    ep_plane = b[:, :, 17]
    if float(np.max(ep_plane)) > 0.5:
        r, c = np.unravel_index(int(np.argmax(ep_plane)), (BOARD_HEIGHT, BOARD_WIDTH))
        ep_sq = int(r * 8 + c)
    out[33] = np.uint8(ep_sq)
    return out


def packed_to_board_tensor(packed) -> torch.Tensor:
    """uint8 (N, PACKED_BOARD_LEN) or (PACKED_BOARD_LEN,) -> float32 torch.Tensor (N, 8, 8, 18).

    Accepts numpy arrays or torch.Tensors. Always returns a batched tensor.
    Fully vectorized (no per-sample/per-square Python loop) so it is cheap on the
    training hot path. Behaviour is bit-identical to the scalar reference; the
    round-trip and real-row tests guard the equivalence.
    """
    if isinstance(packed, torch.Tensor):
        p = packed.detach().cpu().numpy().astype(np.uint8)
    else:
        p = np.asarray(packed, dtype=np.uint8)

    if p.ndim == 1:
        p = p[np.newaxis, :]

    n = p.shape[0]
    if p.shape[1] != PACKED_BOARD_LEN:
        raise ValueError(f"expected packed length {PACKED_BOARD_LEN}, got {p.shape[1]}")

    # Unpack 64 nibbles: two per byte, little-nibble first (even sq -> low, odd sq -> high).
    nib_bytes = p[:, :32]
    nibs = np.empty((n, 64), dtype=np.uint8)
    nibs[:, 0::2] = nib_bytes & 0x0F
    nibs[:, 1::2] = nib_bytes >> 4
    if np.any(nibs > 12):
        raise ValueError(f"invalid packed nibble {int(nibs[nibs > 12][0])} (must be 0..12)")

    # Flat (N, 64, C) in square-major C order; square sq = rank*8 + file, so a final
    # reshape to (N, 8, 8, C) yields [n, rank, file, plane] exactly as the reference.
    out = np.zeros((n, 64, BOARD_CHANNELS), dtype=np.float32)

    # Piece planes 0..11: one-hot scatter at occupied squares (nibble 1..12 -> plane 0..11).
    occ = nibs > 0
    if occ.any():
        ni, si = np.nonzero(occ)
        out[ni, si, nibs[ni, si].astype(np.int64) - 1] = 1.0

    # Meta byte 32: side-to-move (plane 12) + castling (planes 13..16), broadcast over squares.
    meta = p[:, 32].astype(np.int64)[:, None]  # (n, 1)
    out[:, :, 12] = ((meta & 1) > 0).astype(np.float32)
    out[:, :, 13] = ((meta & 2) > 0).astype(np.float32)
    out[:, :, 14] = ((meta & 4) > 0).astype(np.float32)
    out[:, :, 15] = ((meta & 8) > 0).astype(np.float32)
    out[:, :, 16] = ((meta & 16) > 0).astype(np.float32)

    # Byte 33: en-passant target square (255 = none).
    ep = p[:, 33].astype(np.int64)
    has_ep = ep < 64
    if has_ep.any():
        ne = np.nonzero(has_ep)[0]
        out[ne, ep[ne], 17] = 1.0

    return torch.from_numpy(out.reshape(n, BOARD_HEIGHT, BOARD_WIDTH, BOARD_CHANNELS))


def legal_mask_float_to_u64(mask: np.ndarray) -> np.uint64:
    """(64,) float32 with values 0/1 -> bitboard."""
    m = np.asarray(mask, dtype=np.float32).reshape(64)
    bits = 0
    for i in range(64):
        if float(m[i]) > 0.5:
            bits |= 1 << i
    return np.uint64(bits)


def u64_to_legal_mask_float(bits) -> np.ndarray:
    """Bitboard -> (64,) float32 0/1 matching legal_*_mask output."""
    b = int(bits)
    out = np.zeros(64, dtype=np.float32)
    for i in range(64):
        if (b >> i) & 1:
            out[i] = 1.0
    return out


def legal_masks_to_u64(from_mask: np.ndarray, to_mask: np.ndarray) -> tuple[np.uint64, np.uint64]:
    return legal_mask_float_to_u64(from_mask), legal_mask_float_to_u64(to_mask)


def u64_pair_to_masks(from_u64, to_u64) -> tuple[np.ndarray, np.ndarray]:
    return u64_to_legal_mask_float(from_u64), u64_to_legal_mask_float(to_u64)
