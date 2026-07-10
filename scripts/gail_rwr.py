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


def gfeat(model, packed, bs=1024):
    """[CLS ++ mean-sq] global features (frozen encoder). packed torch uint8 (N,34) -> (N,512)."""
    out = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(packed), bs):
            c, s = model.encode(packed[i:i+bs].to(DEV), hist=None)
            out.append(torch.cat([c.float(), s.float().mean(1)], 1).cpu())
    return torch.cat(out)


@torch.no_grad()
def collect(model, head, B, n_plies, band, seed):
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
        frm = torch.multinomial(torch.softmax(fl, -1), 1, generator=g).squeeze(1)
        tmask = legal_to_mask(lam, frm, color); tmask[no_legal, 0] = True
        tl = _san(head.to_logits(sq, frm, cls).float().masked_fill(~tmask, -1e9))
        to = torch.multinomial(torch.softmax(tl, -1), 1, generator=g).squeeze(1)
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
    packed = buf["packed"]; frm = buf["frm"].to(DEV); to = buf["to"].to(DEV)
    fmask = buf["fmask"].to(DEV); tmask = buf["tmask"].to(DEV)
    # cache frozen features once
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        cls, sq = [], []
        for i in range(0, len(packed), 1024):
            c, s = model.encode(packed[i:i+1024].to(DEV), hist=None); cls.append(c.float()); sq.append(s.float())
        cls = torch.cat(cls); sq = torch.cat(sq)
        Gn = gfeat(model, buf["nxt"]).to(DEV)
        r = D(Gn).squeeze(1)                                              # reward = D logit (GRADED)
        A = ((r - r.mean()) / (r.std() + 1e-6)).clamp(-3, 3)               # advantage
        w = (torch.softmax(A, 0) * len(A)).detach()   # RWR weights: NON-NEGATIVE, mean 1 (only pulls
        #                                               good moves UP; never drives logp to -inf)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    N = len(packed); idx = torch.arange(N, device=DEV)
    for _ in range(K):
        perm = idx[torch.randperm(N)]
        for i in range(0, N, 2048):
            b = perm[i:i+2048]
            fl = _san(head.from_logits(sq[b], cls[b])); logp_f = _mlogp(fl, fmask[b], frm[b])
            tl = _san(head.to_logits(sq[b], frm[b], cls[b])); logp_t = _mlogp(tl, tmask[b], to[b])
            rwr = -(w[b] * (logp_f + logp_t)).mean()
            with torch.no_grad():
                rfl = _san(ref.from_logits(sq[b], cls[b])); rtl = _san(ref.to_logits(sq[b], frm[b], cls[b]))
            kl = (_kl(fl, rfl, fmask[b]) + _kl(tl, rtl, tmask[b])).mean()
            loss = rwr + beta * kl
            if not torch.isfinite(loss):        # skip any nan/inf batch (don't poison the head)
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
        buf, pol_states = collect(model, head, a.B, a.plies, a.band, seed=100 + it)
        rng = np.random.default_rng(it)
        hsel = np.sort(rng.choice(hidx_h, min(len(pol_states), len(hidx_h)), replace=False))
        Gh = gfeat(model, torch.from_numpy(f["packed_pre"][hsel].astype(np.int64)).to(torch.uint8))
        Gp = gfeat(model, pol_states)
        D, au = train_disc(Gh, Gp, seed=it)
        beta = a.beta0                                     # constant strong leash (prevents collapse)
        mr, astd = rwr_update(model, head, ref, buf, D, beta, a.K, a.lr)
        print(f"   {it:3d}   {au:.3f}   {mr:+.3f}   (beta={beta:.2f}, |moves|={len(buf['packed'])})", flush=True)
    # final eval: fresh policy vs human
    buf, pol_states = collect(model, head, a.B, a.plies, a.band, seed=999)
    hsel = np.sort(np.random.default_rng(9).choice(hidx_h, min(len(pol_states), len(hidx_h)), replace=False))
    _, au_final = train_disc(gfeat(model, torch.from_numpy(f["packed_pre"][hsel].astype(np.int64)).to(torch.uint8)),
                             gfeat(model, pol_states), seed=9)
    print(f"\n  FINAL held-out disc AUC (human vs RWR policy): {au_final:.3f}")


if __name__ == "__main__":
    main()
