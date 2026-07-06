"""PGN-replay top-1 move-match: Maia-3-23M vs our big MultiBandPolicy, per rating band.

Samples real 2025-05 rapid (tc 600+0) positions from the lichess PGN, balanced per 100-wide
rating band (1000..2100), and measures how often each model's top-1 move equals the move actually
played. Each model gets its native inputs:
  - Maia-3: full move history (from board.move_stack) + SelfElo/OppoElo, argmax (nodes=1).
  - ours: 2-ply history + band routing by the side-to-move elo, joint ARGMAX decode
          (P(from) * P(to|from) over legal moves; NOT sampling).

CPU-ONLY by design: a GPU training job owns the only GPU. Launch with CUDA_VISIBLE_DEVICES="" and
our model loads with map_location="cpu". Maia3Bot is already CPU-only.
"""
from __future__ import annotations
import argparse
import io
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import chess
import chess.pgn
import torch
import zstandard

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from style_policy.maia3_bot import Maia3Bot
from style_policy.multiband_bot import board_history
from style_policy.multiband_policy import MultiBandPolicy
from style_policy.board_encode import board_to_packed

BANDS = list(range(1000, 2200, 100))  # 1000,1100,...,2100 (2100 = 2100-2199)
DEFAULT_PGN = "/mnt/eloquence_bulk/databases/lichess_db_standard_rated_2025-05_tc_600_0.pgn.zst"
DEFAULT_CKPT = ("style_policy_checkpoints/multiband_history_128M_big/"
                "multiband_history_128M_big.pt")


def _band_of(self_elo: int) -> int | None:
    """Map a side-to-move elo to its band, or None if outside our band range."""
    if self_elo < 1000 or self_elo >= 2200:
        return None
    return int(min(2100, max(1000, (self_elo // 100) * 100)))


def sample_positions(pgn_path, per_band, min_ply, seed, max_per_game=3):
    """Iterate games, collecting up to `per_band` balanced samples per band.

    Each sample carries a board copied WITH its move stack (so both models see history),
    the side-to-move / opponent elos, the band, and the actual move played (uci)."""
    rng = random.Random(seed)
    buckets: dict[int, list] = {b: [] for b in BANDS}
    need = set(BANDS)

    dctx = zstandard.ZstdDecompressor()
    with open(pgn_path, "rb") as raw:
        reader = dctx.stream_reader(raw)
        text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        while need:
            game = chess.pgn.read_game(text)
            if game is None:
                break  # PGN exhausted
            try:
                white_elo = int(game.headers.get("WhiteElo", ""))
                black_elo = int(game.headers.get("BlackElo", ""))
            except (ValueError, TypeError):
                continue

            board = game.board()
            taken_this_game = 0
            for p, move in enumerate(game.mainline_moves()):
                if p < min_ply:
                    board.push(move)
                    continue
                if taken_this_game >= max_per_game:
                    board.push(move)
                    continue
                self_elo = white_elo if board.turn == chess.WHITE else black_elo
                oppo_elo = black_elo if board.turn == chess.WHITE else white_elo
                band = _band_of(self_elo)
                if band is None:
                    board.push(move)
                    continue
                if len(buckets[band]) >= per_band:
                    board.push(move)
                    continue
                # Decorrelate within a game: sample this ply with a modest probability.
                if rng.random() < 0.5:
                    buckets[band].append({
                        "board": board.copy(stack=True),
                        "self_elo": int(self_elo),
                        "oppo_elo": int(oppo_elo),
                        "band": band,
                        "actual": move.uci(),
                    })
                    taken_this_game += 1
                    if len(buckets[band]) >= per_band:
                        need.discard(band)
                board.push(move)

    samples = [s for b in BANDS for s in buckets[b]]
    counts = {b: len(buckets[b]) for b in BANDS}
    return samples, counts


def _load_our_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu")
    arch = ck["architecture"]
    model = MultiBandPolicy.from_config(arch)
    model.load_state_dict(ck["model"])
    model.to("cpu").eval()
    n_ply = int(arch.get("n_history_ply", 0)) if arch.get("use_last_move") else 0
    return model, arch, n_ply


@torch.no_grad()
def our_top1(model, n_ply, board, self_elo):
    """Our model's top-1 move via joint ARGMAX: max over legal moves of logP(from)+logP(to|from)."""
    packed = torch.from_numpy(board_to_packed(board)[None])
    hist = None
    if n_ply:
        hf, ht, hc = board_history(board, n_ply)
        hist = (torch.tensor([hf]), torch.tensor([ht]), torch.tensor([hc]))
    cls, squares = model.encode(packed, hist=hist)
    head = model.heads[int(model.head_index(torch.tensor([self_elo])).item())]

    # Dedupe legal moves by (from,to), preferring the queen promo (heads don't model promo piece).
    by_ft: dict = {}
    for m in board.legal_moves:
        k = (m.from_square, m.to_square)
        if k not in by_ft or m.promotion == chess.QUEEN:
            by_ft[k] = m
    moves = list(by_ft.values())
    if not moves:
        return None
    if len(moves) == 1:
        return moves[0].uci()

    froms = sorted({f for f, _ in by_ft})
    fidx = {f: i for i, f in enumerate(froms)}
    logp_from = torch.log_softmax(head.from_logits(squares, cls)[0][froms], dim=-1)
    tl = head.to_logits(squares.expand(len(froms), -1, -1),
                        torch.tensor(froms), cls.expand(len(froms), -1))
    tos_by_from: dict = {}
    for (f, t) in by_ft:
        tos_by_from.setdefault(f, []).append(t)
    logp_to = {f: torch.log_softmax(tl[fidx[f]][tos_by_from[f]], dim=-1) for f in froms}

    best_mv, best_score = None, float("-inf")
    for m in moves:
        f = m.from_square
        score = float(logp_from[fidx[f]]) + float(logp_to[f][tos_by_from[f].index(m.to_square)])
        if score > best_score:
            best_score, best_mv = score, m
    return best_mv.uci()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--maia3-model", default="maia3-23m")
    ap.add_argument("--our-ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--pgn", default=DEFAULT_PGN)
    ap.add_argument("--per-band", type=int, default=200)
    ap.add_argument("--min-ply", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"[1/4] Sampling up to {args.per_band}/band from {args.pgn} "
          f"(min_ply={args.min_ply}, seed={args.seed}) ...", flush=True)
    samples, counts = sample_positions(args.pgn, args.per_band, args.min_ply, args.seed)
    print(f"      got {len(samples)} samples total. Per-band counts:", flush=True)
    for b in BANDS:
        flag = "" if counts[b] >= args.per_band else "  (UNDER-FILLED)"
        print(f"        {b}: {counts[b]}{flag}", flush=True)
    if not samples:
        print("No samples collected; aborting.", file=sys.stderr)
        sys.exit(1)

    print(f"[2/4] Loading our model ({args.our_ckpt}) on CPU ...", flush=True)
    model, arch, n_ply = _load_our_model(args.our_ckpt)
    print(f"      bands={arch.get('bands')} n_ply={n_ply}", flush=True)

    print("[3/4] Scoring our model (joint argmax, CPU) ...", flush=True)
    our_hits = defaultdict(int)
    band_n = defaultdict(int)
    for i, s in enumerate(samples):
        band_n[s["band"]] += 1
        pred = our_top1(model, n_ply, s["board"], s["self_elo"])
        s["our_pred"] = pred
        if pred == s["actual"]:
            our_hits[s["band"]] += 1
        if (i + 1) % 200 == 0:
            print(f"      ours {i + 1}/{len(samples)}", flush=True)

    print(f"[4/4] Scoring Maia-3 ({args.maia3_model}, nodes=1 argmax, CPU) ...", flush=True)
    first = samples[0]
    bot = Maia3Bot(model=args.maia3_model, self_elo=first["self_elo"], oppo_elo=first["oppo_elo"])
    maia_hits = defaultdict(int)
    try:
        for i, s in enumerate(samples):
            bot.set_elos(s["self_elo"], s["oppo_elo"])
            mv = bot.choose_move(s["board"])
            pred = mv.uci() if mv is not None else None
            s["maia_pred"] = pred
            if pred == s["actual"]:
                maia_hits[s["band"]] += 1
            if (i + 1) % 200 == 0:
                print(f"      maia3 {i + 1}/{len(samples)}", flush=True)
    finally:
        bot.close()

    def pct(hits, n):
        return 100.0 * hits / n if n else float("nan")

    print("\n" + "=" * 44)
    print(f"{'band':<6}{'N':>5}   {'maia3-23m%':>10}   {'our-big%':>10}")
    print("-" * 44)
    tot_n = tot_maia = tot_our = 0
    rows = {}
    for b in BANDS:
        n = band_n[b]
        m_pct = pct(maia_hits[b], n)
        o_pct = pct(our_hits[b], n)
        rows[b] = {"N": n, "maia3": m_pct, "ours": o_pct}
        print(f"{b:<6}{n:>5}   {m_pct:>10.1f}   {o_pct:>10.1f}")
        tot_n += n
        tot_maia += maia_hits[b]
        tot_our += our_hits[b]
    all_maia = pct(tot_maia, tot_n)
    all_our = pct(tot_our, tot_n)
    print("-" * 44)
    print(f"{'ALL':<6}{tot_n:>5}   {all_maia:>10.1f}   {all_our:>10.1f}")
    print("=" * 44)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "config": {"maia3_model": args.maia3_model, "our_ckpt": args.our_ckpt,
                           "pgn": args.pgn, "per_band": args.per_band, "min_ply": args.min_ply,
                           "seed": args.seed},
                "per_band_counts": counts,
                "rows": rows,
                "overall": {"N": tot_n, "maia3": all_maia, "ours": all_our},
            }, f, indent=2)
        print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
