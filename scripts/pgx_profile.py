"""Profile the GAIL rollout pipeline per ply, batched over B games, all stages:
  1) pgx step (JAX/GPU)   2) pgx->packed convert (torch/GPU, dlpack)
  3) packed->board_tensor CODEC (currently NUMPY/CPU -> host round trip!)
  4) encoder forward (torch/GPU)   5) from+to heads (torch/GPU)
Reveals where rollout time actually goes so we know what to optimize before GAIL."""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")
import sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, jax, jax.numpy as jnp, numpy as np, pgx
from pgx_bridge import pgx_to_packed
from style_policy.multiband_policy import MultiBandPolicy
from style_policy.packed_codec import packed_to_board_tensor

DEV = "cuda"
def sync(): torch.cuda.synchronize()
def j2t(x): return torch.from_dlpack(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="style_policy_checkpoints/multiband_ourdistill/multiband_ourdistill.pt")
    ap.add_argument("--games", type=int, default=2048)
    ap.add_argument("--plies", type=int, default=30)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location=DEV)
    model = MultiBandPolicy.from_config(ck["architecture"]); model.load_state_dict(ck["model"], strict=False)
    model.to(DEV).eval()
    for p in model.parameters(): p.requires_grad_(False)
    if os.environ.get("COMPILE", "1") == "1":
        model.encoder = torch.compile(model.encoder)   # frozen encoder is static -> compile like training
    head = model.heads[len(model.heads)//2]        # a representative mid band
    enc_params = sum(p.numel() for p in model.encoder.parameters())/1e6
    print(f"encoder {ck['architecture']['d_model']}/{ck['architecture']['n_layers']} ({enc_params:.1f}M) | B={a.games}")

    env = pgx.make("chess")
    step = jax.jit(jax.vmap(env.step))
    state = jax.jit(jax.vmap(env.init))(jax.random.split(jax.random.PRNGKey(0), a.games))
    key = jax.random.PRNGKey(1)
    T = {k: 0.0 for k in ["pgx", "convert", "codec_cpu", "encoder", "heads"]}

    def rand_action(st, k):
        return jax.random.categorical(k, jnp.where(st.legal_action_mask, 0.0, -1e9), axis=1)

    for t in range(a.plies + 3):                    # first 3 = warmup (compile/caches)
        warm = t < 3
        # (2) convert pgx state -> packed (GPU, dlpack + converter)
        sync(); t0 = time.time()
        packed = pgx_to_packed(j2t(state._x.color), j2t(state._x.board),
                               j2t(state._x.castling_rights), j2t(state._x.en_passant))
        sync(); t1 = time.time()
        # (3) codec packed -> board_tensor (NUMPY/CPU round trip) + back to GPU
        board = packed_to_board_tensor(packed).to(DEV)
        sync(); t2 = time.time()
        # (4) encoder forward + (5) heads
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            cls, sq = model.encoder(board, hist=None)
            sync(); t3 = time.time()
            fl = head.from_logits(sq, cls=cls); from_sq = fl.argmax(1)
            tl = head.to_logits(sq, from_sq, cls=cls)
            sync(); t4 = time.time()
        # (1) pgx step
        key, sk = jax.random.split(key)
        state = step(state, rand_action(state, sk)); state.legal_action_mask.block_until_ready()
        t5 = time.time()
        if not warm:
            T["convert"] += t1-t0; T["codec_cpu"] += t2-t1; T["encoder"] += t3-t2
            T["heads"] += t4-t3; T["pgx"] += t5-t4

    tot = sum(T.values())
    print(f"\n=== per-ply breakdown (avg over {a.plies} plies, B={a.games}) ===")
    print(f"{'stage':<14}{'ms/ply':>10}{'% ':>8}")
    for k in ["pgx", "convert", "codec_cpu", "encoder", "heads"]:
        ms = 1000*T[k]/a.plies
        print(f"{k:<14}{ms:>10.2f}{100*T[k]/tot:>8.1f}")
    print(f"{'TOTAL':<14}{1000*tot/a.plies:>10.2f}{100:>8.1f}")
    print(f"\nthroughput: {a.games*a.plies/tot:,.0f} policy-steps/s (full loop)")


if __name__ == "__main__":
    main()
