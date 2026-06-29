#!/usr/bin/env python3
"""Rate dedicated per-band heads vs Maia2 at a FIXED temperature.

Tests whether band identity ALONE separates realized strength (1000-head weaker, 1900-head
stronger) more than the elo-conditioning knob did (which the calibration sweep found ~flat,
±100, temperature-dominated). Temperature is held fixed across heads so the comparison isolates
the head, not temperature.
"""
from __future__ import annotations
import argparse, time
from style_policy.band_head import BandHeadBot
from style_policy.opening_book import OpeningBook
from style_policy.maia2_bot import load_maia2, Maia2Bot
from style_policy.rating import mle_rating
from scripts.rate_bot import bot_record_vs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="style_policy_checkpoints/base_64M/base_64M_stage_1.pt")
    ap.add_argument("--head-dir", default="style_policy_checkpoints/band_heads")
    ap.add_argument("--prefix", default="base_64M_band_", help="head file = {head-dir}/{prefix}{band}.pt")
    ap.add_argument("--bands", type=int, nargs="+", default=[1000, 1500, 1900])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--levels", type=int, nargs="+", default=[1100, 1300, 1500, 1700, 1900])
    ap.add_argument("--games-per-level", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--book-dir", default="/mnt/eloquence_bulk/databases/opening_book")
    ap.add_argument("--no-book", dest="book", action="store_false", default=True)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    maia_dev = "gpu" if str(a.device).startswith("cuda") else "cpu"
    model, prep = load_maia2("rapid", device=maia_dev)
    print(f"rating dedicated band heads ({a.head_dir}/{a.prefix}*.pt) on encoder {a.checkpoint}")
    print(f"FIXED temperature {a.temperature}, book={'on' if a.book else 'off'}; "
          f"vs Maia2 rapid {a.levels}, {a.games_per_level} games/level (color-balanced)")
    print(f"\n{'band':>6} {'rating':>7} {'±95%':>5}   per-Maia score [{' '.join(str(L) for L in a.levels)}]")
    t0 = time.time()
    results = []
    for B in a.bands:
        head_path = f"{a.head_dir}/{a.prefix}{B}.pt"
        book = OpeningBook.for_elo(a.book_dir, B) if a.book else None
        bot = BandHeadBot(head_path, checkpoint=a.checkpoint, device=a.device,
                          temperature=a.temperature, seed=a.seed, opening_book=book)
        rows, scores = [], []
        for R in a.levels:
            maia = Maia2Bot(model, prep, self_elo=R, seed=a.seed + R)
            w, d, l = bot_record_vs(bot, maia, a.games_per_level, a.max_plies)
            s = (w + 0.5 * d) / (w + d + l)
            rows.append((R, w + d + l, s)); scores.append(s)
        rating, se = mle_rating(rows)
        results.append((B, rating))
        print(f"{B:>6} {rating:>7.0f} {1.96*se:>5.0f}   " + "  ".join(f"{x:.2f}" for x in scores), flush=True)
    if len(results) >= 2:
        spread = max(r for _, r in results) - min(r for _, r in results)
        print(f"\nrating spread across bands: {spread:.0f} Elo  "
              f"(elo-conditioning knob was ~±100 / ~200 spread, temperature-dominated)")
    print(f"{time.time()-t0:.0f}s total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
