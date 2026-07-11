"""Calibration reference: how strong is a MINIMAL-search Stockfish (NNUE eval, depth 1) vs Maia2?
Contrast with our 1-ply value bot (~1900): if NNUE-1ply is much stronger, the cap is the EVALUATOR
quality (human-WDL vs Stockfish-NNUE), not the 1-ply-lookahead approach."""
from __future__ import annotations
import argparse, sys, os, chess, chess.engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_policy.play import Player
from style_policy.maia2_bot import load_maia2, Maia2Bot
from style_policy.rating import score_ci, mle_rating
from rate_multiband_ladder import bot_record_vs


class StockfishBot(Player):
    def __init__(self, path="/usr/games/stockfish", depth=1, nodes=0):
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.limit = chess.engine.Limit(nodes=nodes) if nodes else chess.engine.Limit(depth=depth)

    def choose_move(self, board):
        return self.engine.play(board, self.limit).move

    def close(self):
        try: self.engine.quit()
        except Exception: pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=1); ap.add_argument("--nodes", type=int, default=0)
    ap.add_argument("--levels", type=int, nargs="+", default=[1500, 1700, 1900])
    ap.add_argument("--games-per-level", type=int, default=20); ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    maia, prep = load_maia2("rapid", device=a.device)
    bot = StockfishBot(depth=a.depth, nodes=a.nodes)
    lim = f"nodes={a.nodes}" if a.nodes else f"depth={a.depth}"
    print(f"Stockfish ({lim}, NNUE) vs Maia2 {a.levels}", flush=True)
    rows, scores = [], []
    for R in a.levels:
        m = Maia2Bot(maia, prep, self_elo=R, seed=a.seed + R)
        w, d, l = bot_record_vs(bot, m, a.games_per_level, a.max_plies)
        sc, _, _ = score_ci(w, d, l); rows.append((R, w + d + l, sc)); scores.append(sc)
        print(f"  vs Maia {R}: {w}-{d}-{l}  score {sc:.2f}", flush=True)
    bot.close()
    rating, se = mle_rating(rows)
    print(f"\n  Stockfish-{lim} Elo: {rating:.0f} ±{1.96*se:.0f}  "
          f"(our 1-ply value bot ~1869; policy bot ~1900)")


if __name__ == "__main__":
    main()
