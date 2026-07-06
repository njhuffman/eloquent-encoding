"""Allie 2022 blitz test-set top-1 move-match: Maia-3-23M vs our big MultiBandPolicy.

Scores top-1 move-match on the EXACT benchmark Maia-3's ~57% is reported on: the Allie
2022 lichess blitz test set. Applies Maia-3's stated filter -- skip the first 10 full moves
(ply >= 20) and drop positions with < 30s remaining on the mover's clock (reconstructed from
the time-control + per-move seconds-spent). Balanced per 100-wide rating band (1000..2100),
mover elo restricted to [1000, 2199].

Maia-3's number is a CALIBRATION CHECK: if we read the set correctly, Maia-3 should land in
the mid-to-high 50s. Each model gets its native inputs (see scripts/eval_maia3_vs_ours.py).

CPU-ONLY by design: a GPU training job owns the only GPU. Launch with CUDA_VISIBLE_DEVICES=""
and our model loads on CPU. Maia3Bot is already CPU-only.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import chess

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Reuse the existing harness rather than reimplementing scoring.
from eval_maia3_vs_ours import (  # noqa: E402
    BANDS,
    DEFAULT_CKPT,
    Maia3Bot,
    _band_of,
    _load_our_model,
    our_top1,
)

DEFAULT_ALLIE_JSONL = "/mnt/eloquence_bulk/allie/lichess-2022-blitz-test/2022-test-annotated.jsonl"


def sample_allie_positions(jsonl_path, per_band, min_ply, min_clock, seed,
                           max_per_game=3, keep_prob=0.5):
    """Balanced per-band sampler over the Allie blitz test JSONL (one record per game).

    For each game we replay moves-uci, tracking each side's clock from the time control and
    moves-seconds (time SPENT per move). BEFORE pushing move i we look at the mover's remaining
    clock and elo, and keep the position iff:
      - i >= min_ply (skip opening),
      - remaining clock >= min_clock,
      - the mover's band is in range and its bucket is not yet full.
    Within-game positions are decorrelated by capping max_per_game and sampling with keep_prob.
    Each kept sample carries a board copied WITH its move stack (both models see history)."""
    rng = random.Random(seed)
    buckets: dict[int, list] = {b: [] for b in BANDS}
    need = set(BANDS)

    with open(jsonl_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not need:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ucis = rec.get("moves-uci", "").split()
            if not ucis:
                continue
            try:
                white_elo = int(rec["white-elo"])
                black_elo = int(rec["black-elo"])
            except (KeyError, ValueError, TypeError):
                continue
            tc = str(rec.get("time-control", ""))
            try:
                base_s, inc_s = (tc.split("+") + ["0"])[:2]
                base = int(base_s)
                inc = int(inc_s)
            except (ValueError, TypeError):
                continue
            secs = rec.get("moves-seconds") or []

            board = chess.Board()
            wc = bc = float(base)
            taken_this_game = 0
            for i, uci in enumerate(ucis):
                white_to_move = board.turn == chess.WHITE
                clk = wc if white_to_move else bc
                mover_elo = white_elo if white_to_move else black_elo
                oppo_elo = black_elo if white_to_move else white_elo
                band = _band_of(mover_elo)

                keep = (
                    band is not None
                    and i >= min_ply
                    and clk >= min_clock
                    and len(buckets[band]) < per_band
                    and taken_this_game < max_per_game
                    and rng.random() < keep_prob
                )
                if keep:
                    buckets[band].append({
                        "board": board.copy(stack=True),
                        "self_elo": int(mover_elo),
                        "oppo_elo": int(oppo_elo),
                        "band": band,
                        "actual": uci,
                    })
                    taken_this_game += 1
                    if len(buckets[band]) >= per_band:
                        need.discard(band)

                # Advance clock (spent this move, then increment) and the board.
                spent = float(secs[i]) if i < len(secs) else 0.0
                if white_to_move:
                    wc = wc - spent + inc
                else:
                    bc = bc - spent + inc
                try:
                    board.push_uci(uci)
                except (ValueError, AssertionError):
                    break  # malformed transcript; stop replaying this game

    samples = [s for b in BANDS for s in buckets[b]]
    counts = {b: len(buckets[b]) for b in BANDS}
    return samples, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allie-jsonl", default=DEFAULT_ALLIE_JSONL)
    ap.add_argument("--maia3-model", default="maia3-23m")
    ap.add_argument("--our-ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--per-band", type=int, default=500)
    ap.add_argument("--min-ply", type=int, default=20,
                    help="skip first N ply (20 = first 10 full moves)")
    ap.add_argument("--min-clock", type=float, default=30.0,
                    help="drop positions with < this many seconds on the mover's clock")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"[1/4] Sampling up to {args.per_band}/band from {args.allie_jsonl} "
          f"(min_ply={args.min_ply}, min_clock={args.min_clock}, seed={args.seed}) ...",
          flush=True)
    samples, counts = sample_allie_positions(
        args.allie_jsonl, args.per_band, args.min_ply, args.min_clock, args.seed)
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
                           "allie_jsonl": args.allie_jsonl, "per_band": args.per_band,
                           "min_ply": args.min_ply, "min_clock": args.min_clock,
                           "seed": args.seed},
                "per_band_counts": counts,
                "rows": rows,
                "overall": {"N": tot_n, "maia3": all_maia, "ours": all_our},
            }, f, indent=2)
        print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
