"""Rate a 1-ply VALUE bot: use the encoder's WDL value head as an EVALUATOR — for each legal move,
evaluate the resulting position and pick the one that leaves the opponent worst off. Tests whether
the human world-model is a strong EVALUATOR even though its POLICY caps at ~human. Compare to the
policy bot (~1869/1951 at band 2100). Value head order = loss/draw/win; value = P(win)-P(loss) from
the side-to-move's perspective, so we minimize the opponent's value in the resulting position."""
from __future__ import annotations
import argparse, sys, os, numpy as np, torch, chess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_policy.multiband_policy import MultiBandPolicy
from style_policy.board_encode import board_to_packed
from style_policy.model_spec import elo_to_bucket
from style_policy.play import Player
from style_policy.maia2_bot import load_maia2, Maia2Bot
from style_policy.rating import score_ci, mle_rating
from rate_multiband_ladder import bot_record_vs


class ValueBot(Player):
    def __init__(self, checkpoint, band, device="cuda", temperature=0.0, seed=0, sf_value_head=None,
                 use_nnue=False):
        ck = torch.load(checkpoint, map_location=device)
        self.model = MultiBandPolicy.from_config(ck["architecture"])
        self.model.load_state_dict(ck["model"], strict=False); self.model.to(device).eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        self.dev = device; self.temp = temperature
        self.use_nnue = use_nnue                                # use the SF-eval head (scalar tanh cp) as evaluator
        if use_nnue and getattr(self.model, "nnue_head", None) is None:
            raise ValueError("checkpoint has no nnue_head; --use-nnue requires the multi-task model")
        n_elo = int(ck["architecture"]["n_elo_buckets"])
        self.eidx = elo_to_bucket(torch.tensor([band]), n_elo).to(device)
        self.g = torch.Generator(device=device).manual_seed(seed)
        self.sf_head = None
        if sf_value_head:                                      # Stockfish-trained value head (elo-free)
            from style_policy.value_head import WDLHead
            h = torch.load(sf_value_head, map_location=device)
            self.sf_head = WDLHead(d_model=h["d_model"], hidden=h["hidden"], elo_dim=0).to(device).eval()
            self.sf_head.load_state_dict(h["value_head"])

    @torch.no_grad()
    def choose_move(self, board):
        moves = list(board.legal_moves)
        if len(moves) == 1: return moves[0]
        for m in moves:                                        # take a forced mate immediately
            board.push(m); mate = board.is_checkmate(); board.pop()
            if mate: return m
        packs = []
        for m in moves:
            board.push(m); packs.append(board_to_packed(board)); board.pop()
        packed = torch.from_numpy(np.stack(packs).astype(np.int64)).to(self.dev)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            cls, _ = self.model.encode(packed, hist=None)
            if self.use_nnue:                                  # SF-eval head: scalar tanh(cp/400), STM-relative
                opp_val = torch.tanh(self.model.nnue_head(cls).squeeze(-1)).float()
            else:
                wdl = (self.sf_head(cls) if self.sf_head is not None
                       else self.model.value_head(cls, elo_idx=self.eidx.expand(len(moves)))).float()
                opp_val = torch.softmax(wdl, -1)[:, 2] - torch.softmax(wdl, -1)[:, 0]
        # opp_val is the value for the resulting mover (opponent); we minimize it.
        score = -opp_val
        if self.temp > 0:
            i = torch.multinomial(torch.softmax(score / self.temp, 0), 1, generator=self.g).item()
        else:
            i = int(score.argmax())
        return moves[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="style_policy_checkpoints/multiband_ourdistill/multiband_ourdistill.pt")
    ap.add_argument("--band", type=int, default=2100); ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--levels", type=int, nargs="+", default=[1500, 1700, 1900])
    ap.add_argument("--games-per-level", type=int, default=20); ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sf-value-head", default="")
    ap.add_argument("--use-nnue", action="store_true", help="evaluate with the SF-eval head (tanh cp)")
    a = ap.parse_args()
    maia, prep = load_maia2("rapid", device=a.device)
    bot = ValueBot(a.ckpt, a.band, device=a.device, temperature=a.temperature, seed=a.seed,
                   sf_value_head=(a.sf_value_head or None), use_nnue=a.use_nnue)
    tag = "SF-eval-head" if a.use_nnue else ("STOCKFISH-value" if a.sf_value_head else "human-value")
    print(f"1-ply {tag} bot (band {a.band}, T={a.temperature}) vs Maia2 {a.levels}", flush=True)
    rows, scores = [], []
    for R in a.levels:
        m = Maia2Bot(maia, prep, self_elo=R, seed=a.seed + R)
        w, d, l = bot_record_vs(bot, m, a.games_per_level, a.max_plies)
        sc, _, _ = score_ci(w, d, l); rows.append((R, w + d + l, sc)); scores.append(sc)
        print(f"  vs Maia {R}: {w}-{d}-{l}  score {sc:.2f}", flush=True)
    rating, se = mle_rating(rows)
    print(f"\n  VALUE-bot Elo: {rating:.0f} ±{1.96*se:.0f}  (policy bot was ~1869@T0.5 / ~1951@T0.3)")


if __name__ == "__main__":
    main()
