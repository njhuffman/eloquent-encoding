"""Proof-of-improvement RL: strengthen the elo-1000-conditioned human head via 1-ply policy gradient,
using the model's OWN SF-eval head as the reward. Frozen encoder; train only the human head + its elo
embedding. REINFORCE advantage = (our value after the sampled move) - (value of the current position),
both from the eval head. No pgx / no self-play rollout. Ceiling ~ the 1-ply-eval bot (~1961), not the
SF-move policy ceiling — this just demonstrates RL moving the weak head upward.

Progress is measured by a head-to-head of the RL'd head vs a FROZEN copy of the elo-1000 baseline."""
from __future__ import annotations
import argparse, sys, os, copy, numpy as np, torch, chess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_policy.flat_policy import FlatMultiTaskPolicy
from style_policy.board_encode import packed_to_board, board_to_packed
from style_policy.model_spec import elo_to_bucket
from style_policy import move_index
from rate_flat_bot import FlatPolicyBot
from rate_multiband_ladder import bot_record_vs

NEG = -1e9
IDX_FROM = torch.tensor(move_index.IDX_FROM); IDX_TO = torch.tensor(move_index.IDX_TO)

def idx_to_move(board, i):
    f, t = int(IDX_FROM[i]), int(IDX_TO[i])
    promo = chess.QUEEN if (board.piece_type_at(f) == chess.PAWN and chess.square_rank(t) in (0, 7)) else None
    mv = chess.Move(f, t, promotion=promo)
    return mv if mv in board.legal_moves else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="style_policy_checkpoints/flat_multitask_128M/flat_multitask_128M.pt")
    ap.add_argument("--data", default="/mnt/eloquence_bulk/databases/wdl_history_128M.h5")
    ap.add_argument("--band", type=int, default=1000); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ent", type=float, default=0.01); ap.add_argument("--eval-every", type=int, default=300)
    ap.add_argument("--h2h-games", type=int, default=60); ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); dev = a.device
    import h5py
    ck = torch.load(a.ckpt, map_location=dev)
    model = FlatMultiTaskPolicy.from_config(ck["architecture"]); model.load_state_dict(ck["model"], strict=False)
    model.to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)
    for p in list(model.human_head.parameters()) + list(model.elo_emb.parameters()): p.requires_grad_(True)
    n_elo = int(ck["architecture"]["n_elo_buckets"])
    eidx0 = elo_to_bucket(torch.tensor([a.band]), n_elo).to(dev)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    idx_from = IDX_FROM.to(dev); idx_to = IDX_TO.to(dev)

    f = h5py.File(a.data, "r"); N = f["packed_pre"].shape[0]
    rng = np.random.default_rng(0)
    torch.save({"architecture": ck["architecture"], "model": model.state_dict()}, "/tmp/rl_base.pt")  # frozen baseline
    base_bot = FlatPolicyBot("/tmp/rl_base.pt", "human", dev, temperature=0.5, band=a.band, seed=7)

    def h2h():
        cur = FlatPolicyBot.__new__(FlatPolicyBot)      # wrap the live model without reloading
        cur.model = model; cur.head = "human"; cur.temp = 0.5; cur.dev = dev
        cur.eidx = eidx0; cur.g = torch.Generator(device=dev).manual_seed(11)
        cur.idx_from = idx_from; cur.idx_to = idx_to
        was = model.training; model.eval()
        w, d, l = bot_record_vs(cur, base_bot, a.h2h_games, 200)
        if was: model.train()
        return w, d, l

    print(f"RL eval-reward on elo-{a.band} human head | steps={a.steps} bs={a.bs} lr={a.lr}", flush=True)
    w, d, l = h2h(); print(f"  step 0: RL vs frozen-baseline {w}-{d}-{l}  score {(w+0.5*d)/(w+d+l):.3f}", flush=True)
    for step in range(1, a.steps + 1):
        rows = np.sort(rng.choice(N, a.bs, replace=False))
        packed = f["packed_pre"][rows]
        boards = [packed_to_board(p.astype(np.uint8)) for p in packed]
        masks = torch.from_numpy(np.stack([move_index.legal_index_mask(b) for b in boards])).to(dev)
        pk = torch.from_numpy(packed.astype(np.int64)).to(dev)
        with torch.no_grad():
            cls, sq = model.encode(pk, hist=None); v_cur = model.eval_value(cls)     # value of current pos (STM=us)
        logits = model.human_logits(cls.detach(), sq.detach(), eidx0.expand(len(pk))).float().masked_fill(~masks, NEG)
        logp_all = torch.log_softmax(logits, 1)
        probs = logp_all.exp()
        a_idx = torch.multinomial(probs, 1).squeeze(1)                               # sampled move index
        logp = logp_all.gather(1, a_idx[:, None]).squeeze(1)
        ent = -(probs * logp_all.clamp(min=-30)).sum(1).mean()
        # apply each sampled move -> resulting position -> eval head reward
        res_pk = []; ok = torch.ones(len(pk), dtype=torch.bool)
        for j in range(len(pk)):
            mv = idx_to_move(boards[j], int(a_idx[j]))
            if mv is None: ok[j] = False; res_pk.append(packed[j]); continue
            boards[j].push(mv); res_pk.append(board_to_packed(boards[j])); boards[j].pop()
        with torch.no_grad():
            cls_r, _ = model.encode(torch.from_numpy(np.stack(res_pk).astype(np.int64)).to(dev), hist=None)
            v_res = model.eval_value(cls_r)                                          # value for opponent (to move)
        reward = -v_res                                                              # our value after the move
        adv = (reward - v_cur).detach()                                             # advantage vs current value
        adv = adv * ok.to(dev).float()
        loss = -(adv * logp).mean() - a.ent * ent
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"  step {step}: adv={float(adv.mean()):+.4f} ent={float(ent):.3f} loss={float(loss):+.4f}", flush=True)
        if step % a.eval_every == 0:
            w, d, l = h2h(); print(f"  [h2h step {step}] RL vs frozen-baseline {w}-{d}-{l}  score {(w+0.5*d)/(w+d+l):.3f}", flush=True)


if __name__ == "__main__":
    main()
