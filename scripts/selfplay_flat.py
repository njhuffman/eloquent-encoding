"""Game-outcome self-play REINFORCE for the flat human head (frozen encoder). Both sides play the
(trainable) elo-conditioned head with temperature; reward = game result (+1 win / -1 loss / 0 draw)
from the mover's perspective; unterminated games adjudicated by material. KL-leash to a frozen copy.
Tests whether game-reward RL can push past the 1-ply-eval ceiling (~1961) toward the SF-move ceiling
(~2333). python-chess self-play (absolute frame, our 1792 space) — slower than pgx but frame-bug-free.

Progress = head-to-head of the RL'd head vs a FROZEN copy of the starting (elo-1000) baseline."""
from __future__ import annotations
import argparse, sys, os, numpy as np, torch, chess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_policy.flat_policy import FlatMultiTaskPolicy
from style_policy.board_encode import board_to_packed
from style_policy.model_spec import elo_to_bucket
from style_policy import move_index
from rate_flat_bot import FlatPolicyBot
from rate_multiband_ladder import bot_record_vs

NEG = -1e9
IDXF = torch.tensor(move_index.IDX_FROM); IDXT = torch.tensor(move_index.IDX_TO)
_SV = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def _move(board, i):
    f, t = int(IDXF[i]), int(IDXT[i])
    promo = chess.QUEEN if (board.piece_type_at(f) == chess.PAWN and chess.square_rank(t) in (0, 7)) else None
    mv = chess.Move(f, t, promotion=promo)
    return mv if mv in board.legal_moves else None


def _material_result(board):                                 # white-perspective approx result in [-1,1]
    s = sum((1 if p.color else -1) * _SV.get(p.piece_type, 0) for p in board.piece_map().values())
    return float(np.tanh(s / 5.0))


def _outcome(board):                                          # white-perspective final result
    o = board.outcome()
    if o is None or o.winner is None:
        return 0.0
    return 1.0 if o.winner == chess.WHITE else -1.0


@torch.no_grad()
def selfplay(model, eidx, B, max_plies, temp, dev, gen):
    boards = [chess.Board() for _ in range(B)]
    result = [None] * B
    recs = []                                                 # (packed, action_idx, mask, gid, mover_color)
    for ply in range(max_plies):
        for i in range(B):
            if result[i] is None and boards[i].is_game_over():
                result[i] = _outcome(boards[i])
        live = [i for i in range(B) if result[i] is None]
        if not live:
            break
        packed = np.stack([board_to_packed(boards[i]) for i in live]).astype(np.int64)
        masks = torch.from_numpy(np.stack([move_index.legal_index_mask(boards[i]) for i in live])).to(dev)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            cls, sq = model.encode(torch.from_numpy(packed).to(dev), hist=None)
            logits = model.human_logits(cls, sq, eidx.expand(len(live))).float().masked_fill(~masks, NEG)
        a = torch.multinomial(torch.softmax(logits / temp, 1), 1, generator=gen).squeeze(1)
        for k, i in enumerate(live):
            mv = _move(boards[i], int(a[k]))
            if mv is None:
                result[i] = 0.0; continue
            recs.append((packed[k], int(a[k]), masks[k].cpu(), i, 0 if boards[i].turn == chess.WHITE else 1))
            boards[i].push(mv)
    n_term = sum(1 for i in range(B) if result[i] is not None and boards[i].is_game_over())
    for i in range(B):
        if result[i] is None:
            result[i] = _material_result(boards[i])           # adjudicate unterminated by material
    buf = {
        "packed": torch.from_numpy(np.stack([r[0] for r in recs])),
        "action": torch.tensor([r[1] for r in recs]),
        "mask": torch.stack([r[2] for r in recs]),
        "reward": torch.tensor([result[r[3]] * (1 if r[4] == 0 else -1) for r in recs], dtype=torch.float32),
    }
    return buf, dict(term=n_term / B, moves=len(recs), draw=np.mean([1.0 if abs(result[i]) < 1e-6 else 0.0 for i in range(B)]))


def rl_update(model, ref_model, eidx, buf, K, lr, ent_c, beta, dev):
    r = buf["reward"].to(dev)
    A = ((r - r.mean()) / (r.std() + 1e-6)).clamp(-3, 3)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    N = len(buf["action"]); mb = 1024; kl_last = 0.0
    for _ in range(K):
        perm = torch.randperm(N)
        for i in range(0, N, mb):
            b = perm[i:i + mb]
            pk = buf["packed"][b].to(dev); msk = buf["mask"][b].to(dev); act = buf["action"][b].to(dev)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                cls, sq = model.encode(pk, hist=None)                         # frozen encoder (shared with ref)
            cls, sq = cls.float(), sq.float()
            logits = model.human_logits(cls, sq, eidx.expand(len(b))).float().masked_fill(~msk, NEG)
            logp_all = torch.log_softmax(logits, 1)
            logp = logp_all.gather(1, act[:, None]).squeeze(1)
            ent = -(logp_all.exp() * logp_all.clamp(min=-30)).sum(1).mean()
            kl = torch.zeros((), device=dev)
            if ref_model is not None and beta > 0:                            # KL leash to the frozen baseline
                with torch.no_grad():
                    rlogp = torch.log_softmax(
                        ref_model.human_logits(cls, sq, eidx.expand(len(b))).float().masked_fill(~msk, NEG), 1)
                kl = (logp_all.exp() * (logp_all - rlogp)).sum(1).mean()
            loss = -(A[b] * logp).mean() - ent_c * ent + beta * kl
            if not torch.isfinite(loss):
                opt.zero_grad(); continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            kl_last = float(kl)
    return float(r.mean()), kl_last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="style_policy_checkpoints/flat_multitask_128M/flat_multitask_128M.pt")
    ap.add_argument("--band", type=int, default=1000); ap.add_argument("--B", type=int, default=192)
    ap.add_argument("--iters", type=int, default=40); ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--temp", type=float, default=0.8); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--K", type=int, default=2); ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--beta", type=float, default=1.0)       # KL-leash strength to the frozen baseline
    ap.add_argument("--eval-every", type=int, default=8); ap.add_argument("--h2h-games", type=int, default=60)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); dev = a.device
    ck = torch.load(a.ckpt, map_location=dev)
    model = FlatMultiTaskPolicy.from_config(ck["architecture"]); model.load_state_dict(ck["model"], strict=False)
    model.to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)
    for p in list(model.human_head.parameters()) + list(model.elo_emb.parameters()): p.requires_grad_(True)
    n_elo = int(ck["architecture"]["n_elo_buckets"]); eidx = elo_to_bucket(torch.tensor([a.band]), n_elo).to(dev)
    global IDXF, IDXT; IDXF, IDXT = IDXF.to(dev), IDXT.to(dev)
    torch.save({"architecture": ck["architecture"], "model": model.state_dict()}, "/tmp/sp_base.pt")
    base = FlatPolicyBot("/tmp/sp_base.pt", "human", dev, temperature=0.5, band=a.band, seed=7)
    ref_model = FlatMultiTaskPolicy.from_config(ck["architecture"])          # frozen KL-leash reference
    ref_model.load_state_dict(torch.load("/tmp/sp_base.pt", map_location=dev)["model"], strict=False)
    ref_model.to(dev).eval()
    for p in ref_model.parameters(): p.requires_grad_(False)
    gen = torch.Generator(device=dev).manual_seed(0)

    def h2h():
        cur = FlatPolicyBot.__new__(FlatPolicyBot)
        cur.model = model; cur.head = "human"; cur.temp = 0.5; cur.dev = dev; cur.eidx = eidx
        cur.g = torch.Generator(device=dev).manual_seed(11); cur.idx_from = IDXF; cur.idx_to = IDXT
        w, d, l = bot_record_vs(cur, base, a.h2h_games, 200)
        return w, d, l

    print(f"self-play RL on elo-{a.band} head | iters={a.iters} B={a.B} temp={a.temp} lr={a.lr} beta={a.beta}", flush=True)
    w, d, l = h2h(); print(f"  iter 0: vs frozen-baseline {w}-{d}-{l}  score {(w+0.5*d)/(w+d+l):.3f}", flush=True)
    for it in range(1, a.iters + 1):
        buf, st = selfplay(model, eidx, a.B, a.max_plies, a.temp, dev, gen)
        rmean, kl = rl_update(model, ref_model, eidx, buf, a.K, a.lr, a.ent, a.beta, dev)
        if it % 2 == 0:
            print(f"  iter {it}: reward={rmean:+.3f} kl={kl:.3f} term={st['term']:.2f} draw={st['draw']:.2f} moves={st['moves']}", flush=True)
        if it % a.eval_every == 0:
            w, d, l = h2h(); print(f"  [h2h iter {it}] vs frozen-baseline {w}-{d}-{l}  score {(w+0.5*d)/(w+d+l):.3f}", flush=True)


if __name__ == "__main__":
    main()
