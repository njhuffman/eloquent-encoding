#!/usr/bin/env python3
"""Rate one bot config by playing color-balanced games vs Maia2 (rapid) at several levels.

Run: python -m scripts.rate_bot [--checkpoint ...] [--elo 1500] [--temperature 0.1]
                                [--levels 1100 1300 1500 1700 1900] [--games-per-level 100]
"""
from __future__ import annotations
import argparse, json, time
from style_policy.play import PolicyBot, play_match
from style_policy.opening_book import OpeningBook
from style_policy.maia2_bot import load_maia2, Maia2Bot
from style_policy.rating import implied_rating, score_ci, mle_rating


def bot_record_vs(bot, maia, games: int, max_plies: int):
    """Bot's (wins, draws, losses) over `games`, split half as White and half as Black."""
    half = games // 2
    a = play_match(bot, maia, half, max_plies=max_plies)            # bot = White
    b = play_match(maia, bot, games - half, max_plies=max_plies)    # bot = Black
    wins = a["white_wins"] + b["black_wins"]
    losses = a["black_wins"] + b["white_wins"]
    draws = a["draws"] + b["draws"]
    return wins, draws, losses


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="style_policy_checkpoints/base_64M/base_64M_stage_1.pt")
    ap.add_argument("--elo", type=int, default=1500)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--book", action="store_true", default=True)
    ap.add_argument("--no-book", dest="book", action="store_false")
    ap.add_argument("--book-dir", default="/mnt/eloquence_bulk/databases/opening_book")
    ap.add_argument("--levels", type=int, nargs="+", default=[1100, 1300, 1500, 1700, 1900])
    ap.add_argument("--games-per-level", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    book = OpeningBook.for_elo(a.book_dir, a.elo) if a.book else None
    if a.book and book is None:
        print(f"(no opening book for elo {a.elo} in {a.book_dir}; playing without book)")
    bot = PolicyBot(a.checkpoint, a.elo, device=a.device, temperature=a.temperature,
                    seed=a.seed, opening_book=book)
    maia_device = "gpu" if str(a.device).startswith("cuda") else "cpu"
    model, prep = load_maia2("rapid", device=maia_device)

    print(f"rating bot: {a.checkpoint} @ elo {a.elo}, T={a.temperature}, book={'on' if book else 'off'}")
    print(f"vs Maia2 rapid {a.levels}, {a.games_per_level} games/level (color-balanced)\n")
    print(f"{'maia':>5} {'W':>4} {'D':>4} {'L':>4} {'score':>6}  implied (95% CI)")
    rows, per_level, t0 = [], [], time.time()
    for R in a.levels:
        maia = Maia2Bot(model, prep, self_elo=R, seed=a.seed + R)
        w, d, l = bot_record_vs(bot, maia, a.games_per_level, a.max_plies)
        score, lo, hi = score_ci(w, d, l)
        imp = implied_rating(score, R)
        rows.append((R, w + d + l, score))
        per_level.append({"level": R, "w": w, "d": d, "l": l, "score": score,
                          "implied": imp, "implied_lo": implied_rating(lo, R),
                          "implied_hi": implied_rating(hi, R)})
        print(f"{R:>5} {w:>4} {d:>4} {l:>4} {score:>6.3f}  {imp:6.0f} "
              f"[{implied_rating(lo, R):.0f}, {implied_rating(hi, R):.0f}]", flush=True)

    rating, se = mle_rating(rows)
    scores = [s for _, _, s in rows]
    mono = all(scores[i] >= scores[i + 1] - 1e-9 for i in range(len(scores) - 1))
    print(f"\n==> bot rating ≈ {rating:.0f}  (95% CI ±{1.96 * se:.0f})   "
          f"[Maia/lichess-rapid scale]   monotonic={mono}")
    print(f"    {sum(n for _, n, _ in rows)} games in {time.time() - t0:.0f}s")
    if a.out:
        json.dump({"checkpoint": a.checkpoint, "elo": a.elo, "temperature": a.temperature,
                   "rating": rating, "se": se, "per_level": per_level}, open(a.out, "w"), indent=2)
        print(f"    wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
