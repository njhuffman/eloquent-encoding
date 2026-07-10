"""pgx (JAX GPU chess) <-> our token pipeline for GAIL self-play.

pgx_to_packed: batched, vectorized converter from pgx GameState arrays to our on-disk packed
format (uint8 [B,34]) that the encoder consumes. All ops are gather/flip/bitwise -> GPU-doable
(runs on CPU or CUDA torch tensors; trivially portable to JAX). Validated by FEN round-trip vs
board_to_packed. Run as a script to validate.

pgx board facts (see memory pgx-integration): board (64,) int32 FILE-MAJOR (idx=file*8+rank),
pieces +-1..6={P,N,B,R,Q,K}, CANONICAL (mover-relative: color=1 => vertical-flip rank + negate).
castling (2,2) bool rows=[mover,opp] cols=[Q,K]; ep file-major in canonical frame, -1=none.
"""
from __future__ import annotations
import torch


def pgx_to_packed(color: torch.Tensor, board: torch.Tensor,
                  castling: torch.Tensor, ep: torch.Tensor) -> torch.Tensor:
    """color (B,), board (B,64), castling (B,2,2) bool, ep (B,) -> packed uint8 (B,34).

    Absolute-frame packed (a1=0 rank-major, nibble 1-6 white / 7-12 black, meta byte, ep byte),
    matching style_policy.board_encode.board_to_packed. Un-canonicalizes pgx's mover-relative board.
    """
    B = board.shape[0]; dev = board.device
    color = color.to(torch.int64)
    b2 = board.to(torch.int64).reshape(B, 8, 8)          # [b, file, rank], canonical signed
    flipped = -torch.flip(b2, dims=[2])                   # black-to-move: vertical-flip rank + negate
    absfr = torch.where(color.view(B, 1, 1) == 0, b2, flipped)   # [b, file, rank] ABSOLUTE signed
    # signed piece -> nibble: +v->v (1-6 white); -v->6+|v| (7-12 black); 0->0
    nib = torch.where(absfr > 0, absfr, torch.where(absfr < 0, 6 - absfr, torch.zeros_like(absfr)))
    nib_rm = nib.transpose(1, 2).reshape(B, 64)           # [file,rank]->[rank,file]-> sq=rank*8+file
    even, odd = nib_rm[:, 0::2], nib_rm[:, 1::2]          # (B,32) each
    packed = torch.zeros(B, 34, dtype=torch.int64, device=dev)
    packed[:, :32] = (even & 0xF) | ((odd & 0xF) << 4)
    # meta byte: bit0 turn(white), bit1 wK, bit2 wQ, bit3 bk, bit4 bq
    ar = torch.arange(B, device=dev)
    wr, br = color, 1 - color                             # white's row = color; black's = 1-color
    c = castling.to(torch.int64)
    meta = (color == 0).to(torch.int64) \
        + c[ar, wr, 1] * 2 + c[ar, wr, 0] * 4 + c[ar, br, 1] * 8 + c[ar, br, 0] * 16
    packed[:, 32] = meta
    # ep byte: un-flip rank when black-to-move, re-index to rank-major python-chess square
    file = torch.div(ep, 8, rounding_mode='floor'); rank_s = ep % 8
    abs_rank = torch.where(color == 0, rank_s, 7 - rank_s)
    py_sq = abs_rank * 8 + file
    packed[:, 33] = torch.where(ep < 0, torch.full_like(ep, 255), py_sq)
    return packed.to(torch.uint8)


def _validate(B: int = 128, plies: int = 30, cap: int = 4000):
    import jax, jax.numpy as jnp, numpy as np
    import pgx
    from style_policy.board_encode import board_to_packed
    env = pgx.make("chess")
    key = jax.random.PRNGKey(0)
    state = jax.jit(jax.vmap(env.init))(jax.random.split(key, B))
    step = jax.jit(jax.vmap(env.step))
    mm = {"pieces": 0, "meta": 0, "ep": 0}; checked = 0; first_bad = None
    for t in range(plies):
        color = np.asarray(state._x.color); board = np.asarray(state._x.board)
        castl = np.asarray(state._x.castling_rights); ep = np.asarray(state._x.en_passant)
        term = np.asarray(state.terminated)
        mine = pgx_to_packed(torch.from_numpy(color), torch.from_numpy(board.astype(np.int64)),
                             torch.from_numpy(castl), torch.from_numpy(ep.astype(np.int64))).numpy()
        for i in range(B):
            if term[i] or checked >= cap:
                continue
            single = jax.tree_util.tree_map(lambda x: x[i], state)
            ref = board_to_packed(_board_from_fen(single._to_fen()))
            checked += 1
            if not np.array_equal(mine[i, :32], ref[:32]): mm["pieces"] += 1; first_bad = first_bad or ("pieces", single._to_fen())
            if mine[i, 32] != ref[32]: mm["meta"] += 1; first_bad = first_bad or ("meta", single._to_fen())
            if mine[i, 33] != ref[33]: mm["ep"] += 1; first_bad = first_bad or ("ep", single._to_fen())
        if checked >= cap:
            break
        key, sk = jax.random.split(key)
        logits = jnp.where(state.legal_action_mask, 0.0, -1e9)
        state = step(state, jax.random.categorical(sk, logits, axis=1))
    print(f"validated {checked} positions | mismatches: {mm}")
    if any(mm.values()):
        print("  first bad:", first_bad)
    else:
        print("  ALL FIELDS MATCH board_to_packed(FEN) — converter correct.")


def _board_from_fen(fen):
    import chess
    return chess.Board(fen)


if __name__ == "__main__":
    _validate()
