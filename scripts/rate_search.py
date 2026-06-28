#!/usr/bin/env python3
"""Rate the expectimax bot at several search depths vs Maia2 (rapid)."""
from __future__ import annotations
import argparse, json, time
from style_policy.opening_book import OpeningBook
from style_policy.maia2_bot import load_maia2, Maia2Bot
from style_policy.rating import mle_rating
from style_policy.search_bot import ExpectimaxBot
from scripts.rate_bot import bot_record_vs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="style_policy_checkpoints/wdl_16M/wdl_16M_stage_1.pt")
    ap.add_argument("--elo", type=int, default=1500)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--depths", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--levels", type=int, nargs="+", default=[1100, 1300, 1500, 1700, 1900])
    ap.add_argument("--games-per-level", type=int, default=25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--book", action="store_true", default=True)
    ap.add_argument("--no-book", dest="book", action="store_false")
    ap.add_argument("--book-dir", default="/mnt/eloquence_bulk/databases/opening_book")
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    maia_dev = "gpu" if str(a.device).startswith("cuda") else "cpu"
    model, prep = load_maia2("rapid", device=maia_dev)
    book = OpeningBook.for_elo(a.book_dir, a.elo) if a.book else None
    print(f"expectimax {a.checkpoint} @ elo {a.elo}, width {a.width}, book={'on' if book else 'off'}")
    print(f"depths {a.depths} vs Maia {a.levels}, {a.games_per_level} games/level\n")
    print(f"{'depth':>5} {'rating':>7} {'±95%':>5}   per-Maia score [{' '.join(str(L) for L in a.levels)}]")
    results, t0 = [], time.time()
    for depth in a.depths:
        bot = ExpectimaxBot(a.checkpoint, a.elo, depth, width=a.width, device=a.device,
                            seed=a.seed, opening_book=book)
        rows, scores = [], []
        for R in a.levels:
            maia = Maia2Bot(model, prep, self_elo=R, seed=a.seed + R)
            w, d, l = bot_record_vs(bot, maia, a.games_per_level, a.max_plies)
            s = (w + 0.5 * d) / (w + d + l)
            rows.append((R, w + d + l, s)); scores.append(s)
        rating, se = mle_rating(rows)
        results.append({"depth": depth, "rating": rating, "se": se, "scores": scores})
        print(f"{depth:>5} {rating:>7.0f} {1.96*se:>5.0f}   " + "  ".join(f"{x:.2f}" for x in scores), flush=True)
    print(f"\n{time.time()-t0:.0f}s total")
    if a.out:
        json.dump({"checkpoint": a.checkpoint, "elo": a.elo, "width": a.width, "results": results},
                  open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
