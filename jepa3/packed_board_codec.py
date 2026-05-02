"""Pack / unpack (8, 8, 18) float32 board tensors matching embedding.board_encoding.board_to_tensor."""

from __future__ import annotations

import numpy as np

from embedding.config import BOARD_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH

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


def packed_to_board_tensor(packed: np.ndarray) -> np.ndarray:
    """uint8 (PACKED_BOARD_LEN,) -> float32 (8, 8, 18)."""
    p = np.asarray(packed, dtype=np.uint8).reshape(-1)
    if p.shape[0] != PACKED_BOARD_LEN:
        raise ValueError(f"expected packed length {PACKED_BOARD_LEN}, got {p.shape[0]}")

    out = np.zeros((BOARD_HEIGHT, BOARD_WIDTH, BOARD_CHANNELS), dtype=np.float32)
    for sq in range(64):
        bi = sq // 2
        nib = int(p[bi] & 0x0F) if sq % 2 == 0 else int(p[bi] >> 4) & 0x0F
        if nib < 0 or nib > 12:
            raise ValueError(f"invalid packed nibble {nib} at sq {sq}")
        if nib == 0:
            continue
        rank = sq // 8
        file = sq % 8
        out[rank, file, nib - 1] = 1.0

    meta = int(p[32])
    if meta & 1:
        out[:, :, 12] = 1.0
    else:
        out[:, :, 12] = 0.0
    if meta & 2:
        out[:, :, 13] = 1.0
    if meta & 4:
        out[:, :, 14] = 1.0
    if meta & 8:
        out[:, :, 15] = 1.0
    if meta & 16:
        out[:, :, 16] = 1.0

    ep_sq = int(p[33])
    if ep_sq < 64:
        r, f = ep_sq // 8, ep_sq % 8
        out[r, f, 17] = 1.0
    return out


def legal_mask_float_to_u64(mask: np.ndarray) -> np.uint64:
    """(64,) float32 with values 0/1 -> bitboard."""
    m = np.asarray(mask, dtype=np.float32).reshape(64)
    bits = 0
    for i in range(64):
        if float(m[i]) > 0.5:
            bits |= 1 << i
    return np.uint64(bits)


def u64_to_legal_mask_float(bits: np.uint64 | int) -> np.ndarray:
    """Bitboard -> (64,) float32 0/1 matching legal_*_mask output."""
    b = int(bits)
    out = np.zeros(64, dtype=np.float32)
    for i in range(64):
        if (b >> i) & 1:
            out[i] = 1.0
    return out


def legal_masks_to_u64(from_mask: np.ndarray, to_mask: np.ndarray) -> tuple[np.uint64, np.uint64]:
    return legal_mask_float_to_u64(from_mask), legal_mask_float_to_u64(to_mask)


def u64_pair_to_masks(from_u64: np.uint64 | int, to_u64: np.uint64 | int) -> tuple[np.ndarray, np.ndarray]:
    return u64_to_legal_mask_float(from_u64), u64_to_legal_mask_float(to_u64)
