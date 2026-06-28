#!/usr/bin/env python3
"""Elo-calibration sweep: rate the bot at each of its elo SETTINGS vs Maia2, to see whether the
elo-conditioning knob tracks realized strength. Prints realized-rating vs set-elo per temperature.
"""
from __future__ import annotations
import argparse, time
from style_policy.play import PolicyBot
from style_policy.opening_book import OpeningBook
from style_policy.maia2_bot import load_maia2, Maia2Bot
from style_policy.rating import mle_rating
from scripts.rate_bot import bot_record_vs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="style_policy_checkpoints/base_64M/base_64M_stage_1.pt")
    ap.add_argument("--bot-elos", type=int, nargs="+", default=[1100, 1300, 1500, 1700, 1900])
    ap.add_argument("--levels", type=int, nargs="+", default=[1100, 1300, 1500, 1700, 1900])
    ap.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 0.1])
    ap.add_argument("--games-per-level", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--book-dir", default="/mnt/eloquence_bulk/databases/opening_book")
    ap.add_argument("--no-book", dest="book", action="store_false", default=True)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    maia_dev = "gpu" if str(a.device).startswith("cuda") else "cpu"
    model, prep = load_maia2("rapid", device=maia_dev)
    print(f"calibration sweep: {a.checkpoint}", flush=True)
    print(f"bot-elos {a.bot_elos} vs Maia {a.levels}, {a.games_per_level} games/cell, temps {a.temperatures}")
    t0 = time.time()
    for temp in a.temperatures:
        print(f"\n=== temperature {temp}  (book={'on' if a.book else 'off'}) ===")
        print(f"{'set_elo':>7} {'realized':>9} {'±95%':>5}   per-Maia score [{' '.join(str(L) for L in a.levels)}]")
        for be in a.bot_elos:
            book = OpeningBook.for_elo(a.book_dir, be) if a.book else None
            bot = PolicyBot(a.checkpoint, be, device=a.device, temperature=temp, seed=a.seed, opening_book=book)
            rows, scores = [], []
            for R in a.levels:
                maia = Maia2Bot(model, prep, self_elo=R, seed=a.seed + R)
                w, d, l = bot_record_vs(bot, maia, a.games_per_level, a.max_plies)
                s = (w + 0.5 * d) / (w + d + l)
                rows.append((R, w + d + l, s)); scores.append(s)
            rating, se = mle_rating(rows)
            print(f"{be:>7} {rating:>9.0f} {1.96*se:>5.0f}   " + "  ".join(f"{x:.2f}" for x in scores), flush=True)
    print(f"\n{time.time()-t0:.0f}s total", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
