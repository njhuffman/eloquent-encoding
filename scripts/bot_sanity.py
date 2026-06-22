#!/usr/bin/env python3
"""Sanity check for PolicyBot: play elo pairs, confirm stronger beats weaker and white edge."""
from __future__ import annotations
import argparse, time
from style_policy.play import PolicyBot, play_match


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()
    pairs = [(1200, 1200), (1500, 1500), (1800, 1800),
             (1800, 1200), (1200, 1800), (1500, 1200), (1200, 1500)]
    print(f"white_elo black_elo |  white%   draw%  black%   (n={args.games}, T={args.temperature:.1f})")
    t0 = time.time()
    for we, be in pairs:
        w = PolicyBot(args.checkpoint, we, device=args.device, temperature=args.temperature, seed=1)
        b = PolicyBot(args.checkpoint, be, device=args.device, temperature=args.temperature, seed=2)
        r = play_match(w, b, args.games)
        n = r["n"]
        print("  %4d     %4d    |  %5.1f   %5.1f   %5.1f" %
              (we, be, 100 * r["white_wins"] / n, 100 * r["draws"] / n, 100 * r["black_wins"] / n), flush=True)
    print("done (%.0fs)" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
