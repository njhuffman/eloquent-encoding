#!/usr/bin/env python3
"""Bot-Elo strength ladder for the joint MultiBandPolicy: for each conditioning band, play
color-balanced games vs Maia2 (rapid) at several levels and compute the bot's implied Elo
(MLE over levels). Answers: does band-B play ~B Elo, and is the dial monotonic?

Bots feed last-n-ply history during play (MultiBandBot). Temperature 1.0 = sample the band's
learned distribution faithfully (most human-like); lower temp sharpens (stronger)."""
from __future__ import annotations
import argparse, json, time
from style_policy.multiband_bot import MultiBandBot
from style_policy.maia2_bot import load_maia2, Maia2Bot
from style_policy.opening_book import OpeningBook
from style_policy.rating import implied_rating, score_ci, mle_rating
from style_policy.play import play_match


def bot_record_vs(bot, maia, games, max_plies):
    half = games // 2
    a = play_match(bot, maia, half, max_plies=max_plies)
    b = play_match(maia, bot, games - half, max_plies=max_plies)
    return (a["white_wins"] + b["black_wins"], a["draws"] + b["draws"], a["black_wins"] + b["white_wins"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="style_policy_checkpoints/multiband_history_128M_big/multiband_history_128M_big.pt")
    ap.add_argument("--bands", type=int, nargs="+", default=[1100, 1300, 1500, 1700, 1900])
    ap.add_argument("--levels", type=int, nargs="+", default=[1300, 1500, 1700])
    ap.add_argument("--games-per-level", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--book-dir", default="/mnt/eloquence_bulk/databases/opening_book")
    ap.add_argument("--no-book", dest="book", action="store_false", default=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    maia_device = "gpu" if str(a.device).startswith("cuda") else "cpu"
    model, prep = load_maia2("rapid", device=maia_device)
    print(f"ladder: {a.checkpoint}  T={a.temperature}  vs Maia2 {a.levels}  {a.games_per_level} games/level\n")
    print(f"{'band':>5} {'Elo':>6} {'±95%':>5}  {'target-Δ':>8}  monotone-in-band")
    results = []
    t0 = time.time()
    for B in a.bands:
        book = OpeningBook.for_elo(a.book_dir, B) if a.book else None
        bot = MultiBandBot(a.checkpoint, B, device=a.device, temperature=a.temperature,
                           seed=a.seed, opening_book=book)
        rows, scores = [], []
        for R in a.levels:
            maia = Maia2Bot(model, prep, self_elo=R, seed=a.seed + R)
            w, d, l = bot_record_vs(bot, maia, a.games_per_level, a.max_plies)
            score, _, _ = score_ci(w, d, l)
            rows.append((R, w + d + l, score)); scores.append(score)
        rating, se = mle_rating(rows)
        mono = all(scores[i] >= scores[i + 1] - 1e-9 for i in range(len(scores) - 1))
        results.append({"band": B, "elo": rating, "se": se, "delta": rating - B,
                        "scores": scores, "levels": a.levels})
        print(f"{B:>5} {rating:>6.0f} {1.96*se:>5.0f}  {rating-B:>+8.0f}  {mono}", flush=True)

    elos = [r["elo"] for r in results]
    ladder_mono = all(elos[i] <= elos[i + 1] + 1e-9 for i in range(len(elos) - 1))
    print(f"\nladder monotonic across bands: {ladder_mono}   ({sum(a.games_per_level*len(a.levels) for _ in a.bands)} games, {time.time()-t0:.0f}s)")
    if a.out:
        json.dump(results, open(a.out, "w"), indent=2); print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
