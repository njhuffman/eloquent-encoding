"""our<->pgx action bridge for GAIL self-play (the reverse of pgx_bridge's board converter).

pgx action = AlphaZero label: label = from*73 + plane, squares FILE-MAJOR (idx=file*8+rank),
in pgx's CANONICAL frame (rank-flipped for black-to-move). We use pgx's own TO_PLANE/FROM_PLANE
tables so the AZ scheme is never reimplemented.

our_move_to_pgx_action: our (from,to) in ABSOLUTE python-chess rank-major squares -> pgx label.
  (Queen promotion + all normal moves; underpromotions -> queen plane, since our factored policy
   always queens. Every legal chess move — incl. castling/ep/double-push — is queen+knight geometry,
   so TO_PLANE covers it.)
Vectorized torch (runs on GPU); validated by round-trip vs python-chess legal moves.
"""
from __future__ import annotations
import numpy as np, torch
from pgx._src.games.chess import TO_PLANE as _TO_PLANE_NP, FROM_PLANE as _FROM_PLANE_NP

_TO_PLANE = torch.from_numpy(np.asarray(_TO_PLANE_NP, dtype=np.int64))     # (64,64) from,to -> plane (-1)
_FROM_PLANE = torch.from_numpy(np.asarray(_FROM_PLANE_NP, dtype=np.int64)) # (64,73) from,plane -> to (-1)


def _canon_rank(rank, color):   # absolute rank <-> canonical rank: flip when black-to-move
    return torch.where(color == 1, 7 - rank, rank)


def our_move_to_pgx_action(from_abs, to_abs, color):
    """(B,) python-chess squares (rank*8+file) + color -> (label (B,), valid (B,) bool)."""
    tp = _TO_PLANE.to(from_abs.device)
    ff, fr = from_abs % 8, from_abs // 8            # file, rank (rank-major -> components)
    tf, tr = to_abs % 8, to_abs // 8
    from_can = ff * 8 + _canon_rank(fr, color)      # -> pgx canonical file-major idx
    to_can   = tf * 8 + _canon_rank(tr, color)
    plane = tp[from_can, to_can]
    label = from_can * 73 + plane
    return label, plane >= 0


def pgx_action_to_our_move(label, color):
    """(B,) pgx label + color -> (from_abs (B,), to_abs (B,), valid (B,) bool) in python-chess squares."""
    fp = _FROM_PLANE.to(label.device)
    from_can, plane = label // 73, label % 73
    to_can = fp[from_can, plane]
    valid = to_can >= 0
    to_can_s = to_can.clamp(min=0)
    ff, fr = from_can // 8, from_can % 8            # pgx file-major idx -> file, rank
    tf, tr = to_can_s // 8, to_can_s % 8
    from_abs = _canon_rank(fr, color) * 8 + ff      # un-canonicalize rank, back to rank-major
    to_abs   = _canon_rank(tr, color) * 8 + tf
    return from_abs, to_abs, valid


def _validate(B=128, plies=30, cap=3000):
    import jax, jax.numpy as jnp, numpy as np, pgx, chess
    env = pgx.make("chess")
    state = jax.jit(jax.vmap(env.init))(jax.random.split(jax.random.PRNGKey(0), B))
    step = jax.jit(jax.vmap(env.step))
    key = jax.random.PRNGKey(1)
    dec_ok = fwd_ok = checked = 0
    for t in range(plies):
        color = np.asarray(state._x.color); mask = np.asarray(state.legal_action_mask)
        term = np.asarray(state.terminated)
        for i in range(B):
            if term[i] or checked >= cap:
                continue
            single = jax.tree_util.tree_map(lambda x: x[i], state)
            board = chess.Board(single._to_fen())
            py_pairs = {(m.from_square, m.to_square) for m in board.legal_moves}
            # DECODE: pgx legal labels -> our (from,to) set (dedup promo multiplicity)
            labels = np.nonzero(mask[i])[0]
            col = torch.full((len(labels),), int(color[i]))
            fa, ta, v = pgx_action_to_our_move(torch.from_numpy(labels), col)
            dec_pairs = {(int(f), int(t)) for f, t, ok in zip(fa, ta, v) if ok}
            dec_ok += int(dec_pairs == py_pairs)
            # FORWARD: each python-chess legal move -> label, must be set in pgx mask
            fr = torch.tensor([m.from_square for m in board.legal_moves])
            to = torch.tensor([m.to_square for m in board.legal_moves])
            lab, valid = our_move_to_pgx_action(fr, to, torch.full((len(fr),), int(color[i])))
            fwd_ok += int(bool(valid.all()) and bool(mask[i][lab.numpy()].all()))
            checked += 1
        key, sk = jax.random.split(key)
        state = step(state, jax.random.categorical(sk, jnp.where(state.legal_action_mask, 0.0, -1e9), axis=1))
        if checked >= cap:
            break
    print(f"validated {checked} positions:")
    print(f"  DECODE  (pgx legal labels -> our (from,to) set == python-chess): {dec_ok}/{checked}")
    print(f"  FORWARD (python-chess move -> pgx label is legal):               {fwd_ok}/{checked}")
    print("  BRIDGE OK" if dec_ok == checked == fwd_ok else "  MISMATCH — inspect")


if __name__ == "__main__":
    _validate()
