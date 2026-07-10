"""Policy-driven pgx self-play rollout for GAIL (the stage-b/c engine).

Drives batched pgx games with OUR frozen-encoder policy: pgx state -> packed -> encode ->
from/to heads (masked to legal moves) -> sample -> our_move_to_pgx_action -> pgx step.
The new primitive is GPU legal-move masking: derive legal from-squares and legal to-squares
(given a chosen from) in OUR ABSOLUTE frame from pgx's canonical legal_action_mask.
"""
from __future__ import annotations
import torch
from pgx_action import _FROM_PLANE, our_move_to_pgx_action
from pgx_bridge import pgx_to_packed

_NEG = float("-inf")

# absolute(rank-major) square -> pgx canonical(file-major) square; row 0 white-to-move, row 1 black.
# a = rank*8+file; canonical = file*8 + (rank if white else 7-rank). Self-inverse, so also maps back.
_PERM = torch.zeros(2, 64, dtype=torch.int64)
for _a in range(64):
    _r, _f = _a // 8, _a % 8
    _PERM[0, _a] = _f * 8 + _r
    _PERM[1, _a] = _f * 8 + (7 - _r)


def _perm(color, dev):
    return _PERM.to(dev)[color]                       # (B,64) per-game square permutation


def legal_from_mask(lam, color):
    """lam (B,4672) bool, color (B,) -> legal from-square mask (B,64) in ABSOLUTE squares."""
    B = lam.shape[0]
    from_can = lam.view(B, 64, 73).any(-1)            # (B,64) legal froms in canonical squares
    return torch.gather(from_can, 1, _perm(color, lam.device))   # -> absolute


def legal_to_mask(lam, from_abs, color):
    """legal to-square mask (B,64) ABSOLUTE for the chosen from_abs (B,)."""
    B = lam.shape[0]; dev = lam.device
    from_can = torch.gather(_perm(color, dev), 1, from_abs[:, None]).squeeze(1)   # abs->can
    planes = lam.view(B, 64, 73)[torch.arange(B, device=dev), from_can]           # (B,73) legal planes
    to_can = _FROM_PLANE.to(dev)[from_can]                                        # (B,73) to per plane
    valid = planes & (to_can >= 0)
    to_can_mask = torch.zeros(B, 64, device=dev)
    to_can_mask.scatter_reduce_(1, to_can.clamp(min=0), valid.float(), reduce="amax", include_self=True)
    return torch.gather(to_can_mask > 0, 1, _perm(color, dev))                    # -> absolute


def _validate_masking(B=128, plies=25, cap=2500):
    import jax, jax.numpy as jnp, numpy as np, pgx, chess
    env = pgx.make("chess")
    state = jax.jit(jax.vmap(env.init))(jax.random.split(jax.random.PRNGKey(0), B))
    step = jax.jit(jax.vmap(env.step)); key = jax.random.PRNGKey(1)
    fm_ok = tm_ok = checked = 0
    for t in range(plies):
        lam = torch.from_numpy(np.asarray(state.legal_action_mask))
        color = torch.from_numpy(np.asarray(state._x.color).astype(np.int64))
        term = np.asarray(state.terminated)
        fmask = legal_from_mask(lam, color)
        for i in range(B):
            if term[i] or checked >= cap: continue
            single = jax.tree_util.tree_map(lambda x: x[i], state)
            board = chess.Board(single._to_fen())
            py_from = {m.from_square for m in board.legal_moves}
            fm_ok += int(set(torch.nonzero(fmask[i]).flatten().tolist()) == py_from)
            # to-mask for a random legal from
            if py_from:
                fsq = sorted(py_from)[checked % len(py_from)]
                tmask = legal_to_mask(lam[i:i+1], torch.tensor([fsq]), color[i:i+1])
                py_to = {m.to_square for m in board.legal_moves if m.from_square == fsq}
                tm_ok += int(set(torch.nonzero(tmask[0]).flatten().tolist()) == py_to)
            checked += 1
        key, sk = jax.random.split(key)
        state = step(state, jax.random.categorical(sk, jnp.where(state.legal_action_mask, 0.0, -1e9), axis=1))
        if checked >= cap: break
    print(f"validated {checked} positions:")
    print(f"  legal_from_mask == python-chess froms: {fm_ok}/{checked}")
    print(f"  legal_to_mask   == python-chess tos:   {tm_ok}/{checked}")
    print("  MASKING OK" if fm_ok == checked == tm_ok else "  MISMATCH")


def _j2t(x):
    return torch.from_dlpack(x)

def _t2j(x):
    import jax
    try: return jax.numpy.from_dlpack(x)
    except Exception: import numpy as np; return jax.numpy.asarray(x.detach().cpu().numpy())


@torch.no_grad()
def rollout(ckpt, B=512, n_plies=40, band=1500, dev="cuda", seed=0, collect_every=4):
    """Self-play B pgx games with the frozen-encoder policy (sampled, masked). Returns
    (snapshots list of (n,34) packed, all_legal bool, term_count, plies_done)."""
    import jax, jax.numpy as jnp, numpy as np, pgx
    from style_policy.multiband_policy import MultiBandPolicy
    ck = torch.load(ckpt, map_location=dev)
    model = MultiBandPolicy.from_config(ck["architecture"]); model.load_state_dict(ck["model"], strict=False)
    model.to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)
    head = model.heads[int(model.head_index(torch.tensor([band])).item())]
    env = pgx.make("chess"); step = jax.jit(jax.vmap(env.step))
    state = jax.jit(jax.vmap(env.init))(jax.random.split(jax.random.PRNGKey(seed), B))
    g = torch.Generator(device=dev).manual_seed(seed)
    snaps = []; all_legal = True
    for ply in range(n_plies):
        color = _j2t(state._x.color).to(torch.int64)
        term = _j2t(state.terminated)
        packed = pgx_to_packed(color, _j2t(state._x.board),
                               _j2t(state._x.castling_rights), _j2t(state._x.en_passant))
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            cls, sq = model.encode(packed, hist=None)
        lam = _j2t(state.legal_action_mask)
        fmask = legal_from_mask(lam, color)
        no_legal = ~fmask.any(1); fmask[no_legal, 0] = True          # guard terminated (avoid nan)
        fl = head.from_logits(sq, cls).float().masked_fill(~fmask, _NEG)
        from_abs = torch.multinomial(torch.softmax(fl, -1), 1, generator=g).squeeze(1)
        tmask = legal_to_mask(lam, from_abs, color); tmask[no_legal, 0] = True
        tl = head.to_logits(sq, from_abs, cls).float().masked_fill(~tmask, _NEG)
        to_abs = torch.multinomial(torch.softmax(tl, -1), 1, generator=g).squeeze(1)
        label, valid = our_move_to_pgx_action(from_abs, to_abs, color)
        legal_bit = torch.gather(lam, 1, label.clamp(0, 4671)[:, None]).squeeze(1)
        all_legal &= bool((valid | term)[~no_legal].all() and (legal_bit | term)[~no_legal].all())
        if ply % collect_every == 0:
            snaps.append(packed[~term].cpu())
        label = torch.where(term, torch.zeros_like(label), label.clamp(0, 4671))
        state = step(state, _t2j(label.to(torch.int32)))
    return snaps, all_legal, int(_j2t(state.terminated).sum().item()), n_plies


def _test_rollout():
    import time
    t0 = time.time()
    snaps, ok, nterm, plies = rollout("style_policy_checkpoints/multiband_ourdistill/multiband_ourdistill.pt",
                                      B=512, n_plies=40)
    dt = time.time() - t0
    tot = sum(len(s) for s in snaps)
    print(f"rollout 512 games x {plies} plies in {dt:.1f}s ({512*plies/dt:,.0f} policy-steps/s)")
    print(f"  all actions legal: {ok} | terminated games: {nterm}/512 | collected {tot} snapshot states")
    print("  ROLLOUT OK" if ok else "  ILLEGAL ACTION — inspect")


if __name__ == "__main__":
    import sys
    _test_rollout() if "--rollout" in sys.argv else _validate_masking()
