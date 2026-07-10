"""Benchmark pgx self-play throughput: B parallel games x T plies, random legal actions,
fully compiled with lax.scan (the realistic rollout pattern). Reports env-steps/sec.
Run on CPU (current jaxlib) and again after installing jax[cuda] for the GPU number."""
from __future__ import annotations
import argparse, time
import jax, jax.numpy as jnp
from jax import lax
import pgx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1024)
    ap.add_argument("--plies", type=int, default=50)
    a = ap.parse_args()
    print("jax devices:", jax.devices())
    env = pgx.make("chess")
    step_v = jax.vmap(env.step)
    key = jax.random.PRNGKey(0)
    state0 = jax.jit(jax.vmap(env.init))(jax.random.split(key, a.games))

    def body(state, k):
        logits = jnp.where(state.legal_action_mask, 0.0, -1e9)
        act = jax.random.categorical(k, logits, axis=1)
        return step_v(state, act), None

    @jax.jit
    def rollout(state, keys):
        final, _ = lax.scan(body, state, keys)
        return final

    keys = jax.random.split(jax.random.PRNGKey(1), a.plies)
    rollout(state0, keys).legal_action_mask.block_until_ready()   # warm/compile
    t0 = time.time()
    for _ in range(3):
        out = rollout(state0, keys)
    out.legal_action_mask.block_until_ready()
    dt = (time.time() - t0) / 3
    steps = a.games * a.plies
    print(f"{a.games} games x {a.plies} plies = {steps:,} env-steps in {dt*1000:.0f} ms "
          f"=> {steps/dt:,.0f} env-steps/s  ({a.games/dt:,.0f} game-rollouts/s)")


if __name__ == "__main__":
    main()
