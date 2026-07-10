"""End-to-end GPU bridge demo: pgx (JAX, GPU) --dlpack--> torch (GPU, zero-copy) --> pgx_to_packed
(GPU) --> our packed tensor, staying on-device (no host round-trip). Validates a sample vs FEN."""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import jax, numpy as np, torch, pgx, chess
from pgx_bridge import pgx_to_packed
from style_policy.board_encode import board_to_packed


def j2t(x):
    """jax array (GPU) -> torch tensor (same GPU), zero-copy via dlpack."""
    try:
        return torch.from_dlpack(x)
    except Exception:
        from torch.utils import dlpack as td
        import jax.dlpack as jd
        return td.from_dlpack(jd.to_dlpack(x))


def main():
    B = 256
    env = pgx.make("chess")
    state = jax.jit(jax.vmap(env.init))(jax.random.split(jax.random.PRNGKey(0), B))
    step = jax.jit(jax.vmap(env.step)); key = jax.random.PRNGKey(1)
    for _ in range(12):                       # a few random plies for variety (castling/ep/captures)
        key, sk = jax.random.split(key)
        logits = jax.numpy.where(state.legal_action_mask, 0.0, -1e9)
        state = step(state, jax.random.categorical(sk, logits, axis=1))
    print("jax board device:", state._x.board.devices())

    # --- the bridge: jax GPU arrays -> torch GPU tensors, zero-copy ---
    board_t = j2t(state._x.board)
    color_t = j2t(state._x.color)
    castl_t = j2t(state._x.castling_rights)
    ep_t    = j2t(state._x.en_passant)
    print(f"bridged to torch: board {tuple(board_t.shape)} {board_t.dtype} on {board_t.device}")

    packed = pgx_to_packed(color_t, board_t, castl_t, ep_t)   # runs on GPU
    print(f"packed: {tuple(packed.shape)} {packed.dtype} on {packed.device}  (no host copy)")

    # validate a sample against the FEN oracle
    p_cpu = packed.cpu().numpy(); ok = 0; n = 0
    for i in range(0, B, 11):
        single = jax.tree_util.tree_map(lambda x: x[i], state)
        ref = board_to_packed(chess.Board(single._to_fen()))
        ok += int(np.array_equal(p_cpu[i], ref)); n += 1
    print(f"FEN-validated sample: {ok}/{n} exact matches")
    print("BRIDGE OK" if ok == n and str(packed.device).startswith("cuda") else "CHECK FAILED")


if __name__ == "__main__":
    main()
