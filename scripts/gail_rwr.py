"""GAIL stage (b): reward-weighted regression on the frozen-encoder policy.

Loop per outer iter:
  1. self-play current policy in pgx -> buffer of (state, action, legal masks, next-state)
  2. train discriminator D(state) : human(+) vs policy-rollout(-) on frozen-encoder global features
  3. reward r = log D(next-state); advantage A = standardized r
  4. K reward-weighted-CE updates on the policy head: loss = -A*logpi(a|s) + beta*KL(pi||pi_ref)
  5. monitor: held-out D AUC(human vs current policy) -- SUCCESS = this DROPS toward the floor.

Encoder frozen (distill-256/8). Policy head warm-started from the band head; pi_ref = frozen copy.
"""
from __future__ import annotations
import argparse, copy, os, numpy as np, torch, h5py
import jax, jax.numpy as jnp, pgx
from pgx_bridge import pgx_to_packed
from pgx_action import our_move_to_pgx_action
from pgx_rollout import legal_from_mask, legal_to_mask, _j2t, _t2j, _NEG
from style_policy.multiband_policy import MultiBandPolicy

DEV = "cuda"


def _san(logits):
    """sanitize logits for sampling: nan/posinf -> finite so softmax/multinomial never assert."""
    return torch.nan_to_num(logits, nan=-1e9, posinf=1e4, neginf=-1e9)


def sa_feat(model, packed, frm, to):
    """STATE-ACTION features: [encoder global (512) ++ from one-hot (64) ++ to one-hot (64)] = 640."""
    G = gfeat(model, packed)
    of = torch.nn.functional.one_hot(frm.long(), 64).float()
    ot = torch.nn.functional.one_hot(to.long(), 64).float()
    return torch.cat([G, of, ot], 1)


def human_sa(model, f, pool, n, rng):
    """Sample n human (state, human-move) pairs -> S-A features (positives)."""
    sel = np.sort(rng.choice(pool, min(n, len(pool)), replace=False))
    pk = torch.from_numpy(f["packed_pre"][sel].astype(np.int64)).to(torch.uint8)
    fr = torch.from_numpy(f["from_sq"][sel].astype(np.int64))
    to = torch.from_numpy(f["to_sq"][sel].astype(np.int64))
    return sa_feat(model, pk, fr, to)


def gfeat(model, packed, bs=1024):
    """[CLS ++ mean-sq] global features (frozen encoder). packed torch uint8 (N,34) -> (N,512)."""
    out = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(packed), bs):
            c, s = model.encode(packed[i:i+bs].to(DEV), hist=None)
            out.append(torch.cat([c.float(), s.float().mean(1)], 1).cpu())
    return torch.cat(out)


@torch.no_grad()
def collect(model, head, B, n_plies, band, seed, rt=1.0):
    """Self-play; return buffer dict of active-move tensors + all visited policy states (packed)."""
    env = pgx.make("chess"); step = jax.jit(jax.vmap(env.step))
    state = jax.jit(jax.vmap(env.init))(jax.random.split(jax.random.PRNGKey(seed), B))
    g = torch.Generator(device=DEV).manual_seed(seed)
    recs = []; prev = None; visited = []
    for ply in range(n_plies):
        color = _j2t(state._x.color).to(torch.int64)
        term = _j2t(state.terminated)
        packed = pgx_to_packed(color, _j2t(state._x.board),
                               _j2t(state._x.castling_rights), _j2t(state._x.en_passant))
        if prev is not None:                       # fill previous ply's next-state
            pk, fa, ta, fm, tm, gidx = prev
            recs.append(dict(packed=pk, frm=fa, to=ta, fmask=fm, tmask=tm, nxt=packed[gidx].cpu()))
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            cls, sq = model.encode(packed.to(DEV), hist=None)
        lam = _j2t(state.legal_action_mask)
        fmask = legal_from_mask(lam, color); no_legal = ~fmask.any(1); fmask[no_legal, 0] = True
        fl = _san(head.from_logits(sq, cls).float().masked_fill(~fmask, -1e9))
        frm = torch.multinomial(torch.softmax(fl / rt, -1), 1, generator=g).squeeze(1)
        tmask = legal_to_mask(lam, frm, color); tmask[no_legal, 0] = True
        tl = _san(head.to_logits(sq, frm, cls).float().masked_fill(~tmask, -1e9))
        to = torch.multinomial(torch.softmax(tl / rt, -1), 1, generator=g).squeeze(1)
        act = ~term & ~no_legal
        ai = torch.nonzero(act).flatten()
        if os.environ.get("DBG") and ply < 4:
            print(f"    C ply {ply}: active={int(act.sum())} term={int(term.sum())} "
                  f"no_legal={int(no_legal.sum())} uniq_from={len(set(frm[act].tolist())) if act.any() else 0} "
                  f"uniq_to={len(set(to[act].tolist())) if act.any() else 0}", flush=True)
        if len(ai):
            gi = np.asarray(_j2t(state.terminated).cpu())  # placeholder to keep gidx mapping simple
            prev = (packed[ai].cpu(), frm[ai].cpu(), to[ai].cpu(), fmask[ai].cpu(), tmask[ai].cpu(),
                    ai.cpu().numpy())
            visited.append(packed[ai].cpu())
        else:
            prev = None
        label, _ = our_move_to_pgx_action(frm, to, color)
        label = torch.where(term | no_legal, torch.zeros_like(label), label.clamp(0, 4671))
        state = step(state, _t2j(label.to(torch.int32)))
    buf = {k: torch.cat([r[k] for r in recs]) for k in ["packed", "frm", "to", "fmask", "tmask", "nxt"]}
    return buf, torch.cat(visited)


def train_disc(Gh, Gp, epochs=5, seed=0, target_auc=0.80):
    """human(+) vs policy(-) -> GRADED disc (weak net + weight-decay + early-stop at target_auc,
    so it does NOT saturate to AUC 1.0 -> reward stays informative)."""
    torch.manual_seed(seed)
    X = torch.cat([Gh, Gp]).to(DEV); Y = torch.cat([torch.ones(len(Gh)), torch.zeros(len(Gp))]).to(DEV)
    p = torch.randperm(len(X)); ntr = int(0.8*len(X)); tri, tei = p[:ntr], p[ntr:]
    D = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.ReLU(),
                            torch.nn.Dropout(0.3), torch.nn.Linear(64, 1)).to(DEV)
    opt = torch.optim.AdamW(D.parameters(), lr=1e-3, weight_decay=1e-2)
    def _auc():
        D.eval()
        with torch.no_grad():
            s = D(X[tei]).squeeze(1).cpu().numpy(); y = Y[tei].cpu().numpy()
            o = np.argsort(s); r = np.empty_like(o, float); r[o] = np.arange(1, len(s)+1)
            npos = y.sum(); return (r[y == 1].sum() - npos*(npos+1)/2) / (npos*(len(y)-npos))
    for _ in range(epochs):
        D.train(); pp = tri[torch.randperm(len(tri))]
        for i in range(0, len(tri), 2048):
            b = pp[i:i+2048]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(D(X[b]).squeeze(1), Y[b])
            opt.zero_grad(); loss.backward(); opt.step()
        if _auc() >= target_auc:            # early-stop before saturation
            break
    D.eval()
    return D, _auc()


_NEGF = -1e9   # FINITE mask value: -inf makes where(mask, 0*-inf, .) give NaN GRADIENTS (corrupts head)

def _mlogp(logits, mask, idx):
    lp = torch.log_softmax(logits.masked_fill(~mask, _NEGF), -1)
    return lp.gather(1, idx[:, None]).squeeze(1)

def _kl(logits, ref_logits, mask):
    lp = torch.log_softmax(logits.masked_fill(~mask, _NEGF), -1)
    lq = torch.log_softmax(ref_logits.masked_fill(~mask, _NEGF), -1)
    term = lp.exp() * (lp - lq)                                  # finite everywhere (NEGF finite)
    return torch.where(mask, term, torch.zeros_like(term)).sum(-1)


def rwr_update(model, head, ref, buf, D, beta, K, lr):
    packed = buf["packed"]; frm = buf["frm"]; to = buf["to"]; fmask = buf["fmask"]; tmask = buf["tmask"]
    with torch.no_grad():                                                 # rewards -> RWR weights (once)
        r = D(sa_feat(model, packed, frm, to).to(DEV)).squeeze(1)         # reward = D(s,a) logit (GRADED)
        A = ((r - r.mean()) / (r.std() + 1e-6)).clamp(-3, 3)
        w = (torch.softmax(A, 0) * len(A)).detach().cpu()                 # NON-NEGATIVE, mean 1
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    N = len(packed); idx = torch.arange(N)
    for _ in range(K):
        perm = idx[torch.randperm(N)]
        for i in range(0, N, 1024):
            b = perm[i:i+1024]
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                cls, sq = model.encode(packed[b].to(DEV), hist=None)       # re-encode (frozen) -> no cache
                cls, sq = cls.float(), sq.float()
            fm, tm, fr, t2 = fmask[b].to(DEV), tmask[b].to(DEV), frm[b].to(DEV), to[b].to(DEV)
            fl = _san(head.from_logits(sq, cls)); logp_f = _mlogp(fl, fm, fr)
            tl = _san(head.to_logits(sq, fr, cls)); logp_t = _mlogp(tl, tm, t2)
            rwr = -(w[b].to(DEV) * (logp_f + logp_t)).mean()
            with torch.no_grad():
                rfl = _san(ref.from_logits(sq, cls)); rtl = _san(ref.to_logits(sq, fr, cls))
            kl = (_kl(fl, rfl, fm) + _kl(tl, rtl, tm)).mean()
            loss = rwr + beta * kl
            if not torch.isfinite(loss):
                opt.zero_grad(); continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
    return float(r.mean()), float(A.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="style_policy_checkpoints/multiband_ourdistill/multiband_ourdistill.pt")
    ap.add_argument("--val", default="/mnt/eloquence_bulk/databases/wdl_validation_2025_05.h5")
    ap.add_argument("--band", type=int, default=1500)
    ap.add_argument("--B", type=int, default=256); ap.add_argument("--plies", type=int, default=24)
    ap.add_argument("--outer", type=int, default=6); ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--beta0", type=float, default=0.5)
    ap.add_argument("--beta1", type=float, default=0.15)   # anneal target (looser leash late)
    ap.add_argument("--rollout-temp", type=float, default=1.0)
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location=DEV)
    model = MultiBandPolicy.from_config(ck["architecture"]); model.load_state_dict(ck["model"], strict=False)
    model.to(DEV).eval()
    for p in model.parameters(): p.requires_grad_(False)
    hidx = int(model.head_index(torch.tensor([a.band])).item())
    head = model.heads[hidx]
    for p in head.parameters(): p.requires_grad_(True)
    ref = copy.deepcopy(head).to(DEV).eval()
    for p in ref.parameters(): p.requires_grad_(False)

    f = h5py.File(a.val, "r"); elo = f["elo_to_move"][:]
    hidx_h = np.where((elo >= a.band) & (elo < a.band + 100))[0]

    print(f"=== GAIL RWR (band {a.band}, B={a.B}, plies={a.plies}, outer={a.outer}, K={a.K}) ===", flush=True)
    print("  outer  discAUC   meanR   |  (AUC should DROP toward ~0.55 floor)", flush=True)
    for it in range(a.outer):
        buf, pol_states = collect(model, head, a.B, a.plies, a.band, seed=100 + it, rt=a.rollout_temp)
        rng = np.random.default_rng(it)
        n = len(buf["packed"])
        Gh = human_sa(model, f, hidx_h, n, rng)                       # human (state, human-move) +
        Gp = sa_feat(model, buf["packed"], buf["frm"], buf["to"])     # policy (state, policy-move) -
        D, au = train_disc(Gh, Gp, seed=it)
        frac = it / max(a.outer - 1, 1)
        beta = a.beta0 * (1 - frac) + a.beta1 * frac       # anneal leash down (allow more movement late)
        mr, astd = rwr_update(model, head, ref, buf, D, beta, a.K, a.lr)
        print(f"   {it:3d}   {au:.3f}   {mr:+.3f}   (beta={beta:.2f}, |moves|={len(buf['packed'])})", flush=True)
    # final eval: fresh S-A disc, human(s,a) vs RWR-policy(s,a)
    buf, _ = collect(model, head, a.B, a.plies, a.band, seed=999, rt=a.rollout_temp)
    rng = np.random.default_rng(9); n = len(buf["packed"])
    _, au_final = train_disc(human_sa(model, f, hidx_h, n, rng),
                             sa_feat(model, buf["packed"], buf["frm"], buf["to"]), seed=9)
    print(f"\n  FINAL held-out S-A disc AUC (human vs RWR policy moves): {au_final:.3f}")
    if a.save:
        torch.save({"architecture": ck["architecture"], "model": model.state_dict()}, a.save)
        print(f"  saved RWR policy -> {a.save}")


if __name__ == "__main__":
    main()
