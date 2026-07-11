"""Policy-only self-play RL for STRENGTH (no search). Warm-start one band head from the human
world-model, self-play to termination in pgx, REINFORCE on game outcome (win/draw/loss). Frozen
encoder. Question: how high does a raw policy on a human-move encoder reach?

Reward: +1/-1/0 for the mover's side at game end. Advantage = standardized reward. Update =
-(A*logp) - ent*entropy + beta*KL(head||frozen prior). Reuses the pgx rollout/bridge/masking stack.
"""
from __future__ import annotations
import argparse, copy, os, numpy as np, torch
import jax, jax.numpy as jnp, pgx
from pgx_bridge import pgx_to_packed
from pgx_action import our_move_to_pgx_action
from pgx_rollout import legal_from_mask, legal_to_mask, _j2t, _t2j
from style_policy.multiband_policy import MultiBandPolicy

DEV = "cuda"; _NEGF = -1e9
def _san(x): return torch.nan_to_num(x, nan=-1e9, posinf=1e4, neginf=-1e9)
def _mlogp(logits, mask, idx):
    return torch.log_softmax(logits.masked_fill(~mask, _NEGF), -1).gather(1, idx[:, None]).squeeze(1)
def _entropy(logits, mask):
    lp = torch.log_softmax(logits.masked_fill(~mask, _NEGF), -1)
    return -torch.where(mask, lp.exp()*lp, torch.zeros_like(lp)).sum(-1)
def _kl(logits, ref, mask):
    lp = torch.log_softmax(logits.masked_fill(~mask, _NEGF), -1)
    lq = torch.log_softmax(ref.masked_fill(~mask, _NEGF), -1)
    return torch.where(mask, lp.exp()*(lp-lq), torch.zeros_like(lp)).sum(-1)


@torch.no_grad()
def selfplay(model, head, B, max_plies, seed, rt=1.0):
    """Self-play B games to termination. Returns buffer of (state,action,masks,reward) per move,
    reward = game outcome (+1/-1/0) from the MOVER's perspective, plus game stats."""
    env = pgx.make("chess"); step = jax.jit(jax.vmap(env.step))
    state = jax.jit(jax.vmap(env.init))(jax.random.split(jax.random.PRNGKey(seed), B))
    g = torch.Generator(device=DEV).manual_seed(seed)
    recs = []; result = torch.zeros(B); done = torch.zeros(B, dtype=torch.bool); plies_played = 0
    for ply in range(max_plies):
        term = _j2t(state.terminated).cpu(); color = _j2t(state._x.color).to(torch.int64)
        rew = _j2t(state.rewards).cpu()                       # (B,2) white/black outcome
        newly = term & ~done
        if newly.any(): result[newly] = rew[newly, 0]         # store white's result
        done = done | term
        if bool(done.all()): break
        plies_played = ply + 1
        packed = pgx_to_packed(color, _j2t(state._x.board),
                               _j2t(state._x.castling_rights), _j2t(state._x.en_passant))
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            cls, sq = model.encode(packed.to(DEV), hist=None)
        lam = _j2t(state.legal_action_mask)
        fmask = legal_from_mask(lam, color); no_legal = ~fmask.any(1); fmask[no_legal, 0] = True
        fl = _san(head.from_logits(sq, cls).float().masked_fill(~fmask, -1e9))
        frm = torch.multinomial(torch.softmax(fl / rt, -1), 1, generator=g).squeeze(1)
        tmask = legal_to_mask(lam, frm, color); tmask[no_legal, 0] = True
        tl = _san(head.to_logits(sq, frm, cls).float().masked_fill(~tmask, -1e9))
        to = torch.multinomial(torch.softmax(tl / rt, -1), 1, generator=g).squeeze(1)
        active = (~_j2t(state.terminated) & ~no_legal).cpu()
        ai = torch.nonzero(active).flatten()
        if len(ai):
            recs.append((packed[ai].cpu(), frm[ai].cpu(), to[ai].cpu(), fmask[ai].cpu(),
                         tmask[ai].cpu(), ai.clone(), color[ai].cpu()))
        label, _ = our_move_to_pgx_action(frm, to, color)
        label = torch.where(_j2t(state.terminated) | no_legal, torch.zeros_like(label), label.clamp(0, 4671))
        state = step(state, _t2j(label.to(torch.int32)))
    # assemble buffer; reward = result[game] * (+1 white-mover / -1 black-mover)
    keys = ["packed", "frm", "to", "fmask", "tmask"]
    buf = {k: torch.cat([r[i] for r in recs]) for i, k in enumerate(keys)}
    gid = torch.cat([r[5] for r in recs]); mover = torch.cat([r[6] for r in recs])
    buf["reward"] = result[gid] * (1 - 2 * mover.float())
    stats = dict(term_rate=float(done.float().mean()), draw_rate=float((result[done] == 0).float().mean()),
                 white_win=float((result[done] > 0).float().mean()), n_moves=len(gid), plies=plies_played)
    return buf, stats


def rl_update(model, head, ref, buf, K, lr, ent_coef, beta):
    packed = buf["packed"]; frm = buf["frm"]; to = buf["to"]; fmask = buf["fmask"]; tmask = buf["tmask"]
    r = buf["reward"].to(DEV)
    A = ((r - r.mean()) / (r.std() + 1e-6)).clamp(-3, 3)      # signed advantage (REINFORCE)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    N = len(packed); idx = torch.arange(N)
    for _ in range(K):
        perm = idx[torch.randperm(N)]
        for i in range(0, N, 1024):
            b = perm[i:i+1024]
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                cls, sq = model.encode(packed[b].to(DEV), hist=None); cls, sq = cls.float(), sq.float()
            fm, tm, fr, t2 = fmask[b].to(DEV), tmask[b].to(DEV), frm[b].to(DEV), to[b].to(DEV)
            fl = _san(head.from_logits(sq, cls)); tl = _san(head.to_logits(sq, fr, cls))
            logp = _mlogp(fl, fm, fr) + _mlogp(tl, tm, t2)
            ent = _entropy(fl, fm) + _entropy(tl, tm)
            with torch.no_grad():
                rfl = _san(ref.from_logits(sq, cls)); rtl = _san(ref.to_logits(sq, fr, cls))
            kl = _kl(fl, rfl, fm) + _kl(tl, rtl, tm)
            loss = -(A[b] * logp).mean() - ent_coef * ent.mean() + beta * kl.mean()
            if not torch.isfinite(loss): opt.zero_grad(); continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
    return float(r.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="style_policy_checkpoints/multiband_ourdistill/multiband_ourdistill.pt")
    ap.add_argument("--band", type=int, default=2100); ap.add_argument("--B", type=int, default=256)
    ap.add_argument("--max-plies", type=int, default=200); ap.add_argument("--outer", type=int, default=10)
    ap.add_argument("--K", type=int, default=4); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ent", type=float, default=0.01); ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--rt", type=float, default=1.0); ap.add_argument("--save", default="")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location=DEV)
    model = MultiBandPolicy.from_config(ck["architecture"]); model.load_state_dict(ck["model"], strict=False)
    model.to(DEV).eval()
    for p in model.parameters(): p.requires_grad_(False)
    head = model.heads[int(model.head_index(torch.tensor([a.band])).item())]
    for p in head.parameters(): p.requires_grad_(True)
    ref = copy.deepcopy(head).to(DEV).eval()
    for p in ref.parameters(): p.requires_grad_(False)
    print(f"=== self-play RL (band-head {a.band}, B={a.B}, outer={a.outer}, K={a.K}) ===", flush=True)
    print("  outer  meanR   draw%  wwin%  moves  plies", flush=True)
    for it in range(a.outer):
        buf, st = selfplay(model, head, a.B, a.max_plies, seed=1000 + it, rt=a.rt)
        mr = rl_update(model, head, ref, buf, a.K, a.lr, a.ent, a.beta)
        print(f"   {it:3d}  {mr:+.3f}  {100*st['draw_rate']:5.1f} {100*st['white_win']:5.1f} "
              f"{st['n_moves']:6d}  {st['plies']}", flush=True)
    if a.save:
        torch.save({"architecture": ck["architecture"], "model": model.state_dict()}, a.save)
        print(f"  saved -> {a.save}")


if __name__ == "__main__":
    main()
