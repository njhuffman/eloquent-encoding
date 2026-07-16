"""Rate a bot that plays a FlatMultiTaskPolicy head (sf_move | human) with temperature, vs Maia2.
The SF-move head bot = the searchless "feature ceiling" X: best play easily obtainable from these
features by directly imitating Stockfish's depth-8 best move. Compare to the human-head bot and to
the value/eval bots (~1961). Promotions assumed queen (flat 1792 space drops under-promotions)."""
from __future__ import annotations
import argparse, sys, os, numpy as np, torch, chess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_policy.flat_policy import FlatMultiTaskPolicy
from style_policy.board_encode import board_to_packed
from style_policy.model_spec import elo_to_bucket
from style_policy.play import Player
from style_policy.maia2_bot import load_maia2, Maia2Bot
from style_policy.rating import score_ci, mle_rating
from style_policy import move_index
from rate_multiband_ladder import bot_record_vs


class FlatPolicyBot(Player):
    def __init__(self, ckpt, head, device="cuda", temperature=0.5, band=2000, seed=0):
        ck = torch.load(ckpt, map_location=device)
        self.model = FlatMultiTaskPolicy.from_config(ck["architecture"])
        self.model.load_state_dict(ck["model"], strict=False); self.model.to(device).eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        self.head = head; self.temp = temperature; self.dev = device
        n_elo = int(ck["architecture"]["n_elo_buckets"])
        self.eidx = elo_to_bucket(torch.tensor([band]), n_elo).to(device)
        self.g = torch.Generator(device=device).manual_seed(seed)
        self.idx_from = torch.tensor(move_index.IDX_FROM, device=device)
        self.idx_to = torch.tensor(move_index.IDX_TO, device=device)

    def _to_move(self, board, i):
        f, t = int(self.idx_from[i]), int(self.idx_to[i])
        promo = chess.QUEEN if (board.piece_type_at(f) == chess.PAWN and chess.square_rank(t) in (0, 7)) else None
        return chess.Move(f, t, promotion=promo)

    @torch.no_grad()
    def choose_move(self, board):
        moves = list(board.legal_moves)
        if len(moves) == 1:
            return moves[0]
        packed = torch.from_numpy(board_to_packed(board).astype(np.int64)).unsqueeze(0).to(self.dev)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev == "cuda"):
            cls, squares = self.model.encode(packed, hist=None)
            logits = (self.model.sf_move_logits(cls, squares) if self.head == "sf_move"
                      else self.model.human_logits(cls, squares, self.eidx))[0].float()
        mask = torch.from_numpy(move_index.legal_index_mask(board)).to(self.dev)
        logits = logits.masked_fill(~mask, -1e9)
        if self.temp > 0:
            i = torch.multinomial(torch.softmax(logits / self.temp, 0), 1, generator=self.g).item()
        else:
            i = int(logits.argmax())
        mv = self._to_move(board, i)
        if mv in board.legal_moves:
            return mv
        for j in torch.argsort(logits, descending=True).tolist():   # safety fallback to best legal
            cand = self._to_move(board, j)
            if cand in board.legal_moves:
                return cand
        return moves[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="style_policy_checkpoints/flat_multitask_128M/flat_multitask_128M.pt")
    ap.add_argument("--head", default="sf_move", choices=["sf_move", "human"])
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--band", type=int, default=2000)
    ap.add_argument("--levels", type=int, nargs="+", default=[1500, 1700, 1900, 2100])
    ap.add_argument("--games-per-level", type=int, default=30); ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    maia, prep = load_maia2("rapid", device=a.device)
    bot = FlatPolicyBot(a.ckpt, a.head, a.device, a.temperature, a.band, a.seed)
    tag = f"{a.head} head (T={a.temperature}{', band '+str(a.band) if a.head=='human' else ''})"
    print(f"flat {tag} bot vs Maia2 {a.levels}", flush=True)
    rows = []
    for R in a.levels:
        m = Maia2Bot(maia, prep, self_elo=R, seed=a.seed + R)
        w, d, l = bot_record_vs(bot, m, a.games_per_level, a.max_plies)
        sc, _, _ = score_ci(w, d, l); rows.append((R, w + d + l, sc))
        print(f"  vs Maia {R}: {w}-{d}-{l}  score {sc:.2f}", flush=True)
    rating, se = mle_rating(rows)
    print(f"\n  flat {tag} Elo: {rating:.0f} ±{1.96*se:.0f}", flush=True)


if __name__ == "__main__":
    main()
