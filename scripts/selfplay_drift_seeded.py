"""Attribution control for self-play drift: seed bot AND human from the SAME real human mid-game
positions (ply P, band-matched games), then compare where each goes K plies later. Same seed
distribution => the opening-repertoire confound is removed; any AUC gap is GENUINE forward drift.
Reports discriminator AUC (human-continuation vs bot-continuation) at several horizons + a floor.
"""
from __future__ import annotations
import argparse, io, sys, os, numpy as np, torch, chess, chess.pgn, zstandard
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selfplay_drift import load, feats, disc_auc, _hist_tensors, _cap
from style_policy.board_encode import board_to_packed


def collect_seeds(pgn, band, P, horizons, n_seeds, n_ply):
    maxK = max(horizons); dctx = zstandard.ZstdDecompressor(); seeds = []
    with open(pgn, "rb") as raw:
        text = io.TextIOWrapper(dctx.stream_reader(raw), encoding="utf-8", errors="replace")
        while len(seeds) < n_seeds:
            game = chess.pgn.read_game(text)
            if game is None: break
            try:
                we = int(game.headers.get("WhiteElo", "")); be = int(game.headers.get("BlackElo", ""))
            except (ValueError, TypeError):
                continue
            if not (band <= we < band + 100 and band <= be < band + 100): continue
            ml = list(game.mainline_moves())
            if len(ml) < P + maxK: continue
            board = chess.Board(); recent = deque(maxlen=max(n_ply, 1))
            seed_board = seed_recent = None; humans = {}
            for p, mv in enumerate(ml):
                if p == P:
                    seed_board = board.copy(); seed_recent = list(recent)
                cap = _cap(board, mv); board.push(mv); recent.append((mv.from_square, mv.to_square, cap))
                if seed_board is not None and (p + 1 - P) in horizons:
                    humans[p + 1 - P] = board.copy()
                if seed_board is not None and (p + 1 - P) >= maxK: break
            if seed_board is not None and len(humans) == len(horizons):
                seeds.append((seed_board, seed_recent, humans))
    return seeds


@torch.no_grad()
def rollout(model, head, n_ply, seeds, horizons, dev, gseed=0):
    g = torch.Generator(device=dev).manual_seed(gseed)
    boards = [s[0].copy() for s in seeds]
    recents = [deque(s[1], maxlen=max(n_ply, 1)) for s in seeds]
    snaps = {h: [None] * len(seeds) for h in horizons}; maxK = max(horizons)
    for step in range(maxK):
        act = [i for i in range(len(boards)) if not boards[i].is_game_over()]
        if act:
            packed = np.stack([board_to_packed(boards[i]) for i in act])
            hist = _hist_tensors([recents[i] for i in act], n_ply, dev) if n_ply else None
            cls, sq = model.encode(torch.from_numpy(packed.astype(np.int64)).to(dev), hist=hist)
            for k, i in enumerate(act):
                b = boards[i]; by = {}
                for m in b.legal_moves:
                    key = (m.from_square, m.to_square)
                    if key not in by or m.promotion == chess.QUEEN: by[key] = m
                froms = sorted({f for f, _ in by})
                fl = head.from_logits(sq[k:k+1], cls[k:k+1])[0][froms]
                fi = torch.multinomial(torch.softmax(fl, -1), 1, generator=g).item()
                f = froms[fi]; tos = [t for (ff, t) in by if ff == f]
                tl = head.to_logits(sq[k:k+1], torch.tensor([f], device=dev), cls[k:k+1])[0][tos]
                ti = torch.multinomial(torch.softmax(tl, -1), 1, generator=g).item()
                mv = by[(f, tos[ti])]; recents[i].append((f, tos[ti], _cap(b, mv))); b.push(mv)
        kk = step + 1
        if kk in horizons:
            for i in range(len(boards)): snaps[kk][i] = board_to_packed(boards[i])
    return snaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="style_policy_checkpoints/multiband_history_128M_big/multiband_history_128M_big.pt")
    ap.add_argument("--pgn", default="/mnt/eloquence_bulk/databases/lichess_db_standard_rated_2025-05_tc_600_0.pgn.zst")
    ap.add_argument("--band", type=int, default=1500); ap.add_argument("--seed-ply", type=int, default=10)
    ap.add_argument("--horizons", default="4,8,12"); ap.add_argument("--n-seeds", type=int, default=1500)
    ap.add_argument("--device", default="cuda"); a = ap.parse_args(); dev = a.device
    horizons = [int(x) for x in a.horizons.split(",")]
    model, n_ply = load(a.ckpt, dev)
    head = model.heads[int(model.head_index(torch.tensor([a.band])).item())]
    print(f"collecting seeds (band {a.band}, ply {a.seed_ply}) ...", flush=True)
    seeds = collect_seeds(a.pgn, a.band, a.seed_ply, horizons, a.n_seeds, n_ply)
    print(f"  {len(seeds)} seeds", flush=True)
    print("bot rollout from human seeds ...", flush=True)
    snaps = rollout(model, head, n_ply, seeds, horizons, dev)

    print(f"\n===== SEEDED DRIFT (band {a.band}, seed ply {a.seed_ply}; same openings) =====")
    print("  human-continuation vs BOT-continuation, K plies after the shared seed:")
    for h in horizons:
        hum = np.stack([board_to_packed(s[2][h]) for s in seeds])
        bot = np.stack(snaps[h])
        Fh = feats(model, hum, dev); Fb = feats(model, bot, dev)
        au = disc_auc(Fh, Fb, dev)
        print(f"    +{h:2d} plies: AUC {au:.3f}", flush=True)
    # floor: human continuations at max horizon, split A/B
    hmax = np.stack([board_to_packed(s[2][max(horizons)]) for s in seeds]); Fm = feats(model, hmax, dev)
    nh = len(Fm) // 2
    print(f"    floor (human A/B at +{max(horizons)}): AUC {disc_auc(Fm[:nh], Fm[nh:], dev):.3f}")
    print("  read: AUC near floor => bot's continuations look human (drift was mostly OPENINGS);")
    print("        AUC rising with horizon => genuine midgame drift accumulating")


if __name__ == "__main__":
    main()
