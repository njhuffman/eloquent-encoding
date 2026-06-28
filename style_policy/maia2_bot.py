"""Maia2 as an anchor opponent on the Player seam. Samples from Maia2's move distribution at a given
rating (sampling, not argmax, so it plays at its nominal level)."""
from __future__ import annotations
import random
import chess
from style_policy.play import Player


def load_maia2(type: str = "rapid", device: str = "gpu"):
    """Load a pretrained Maia2 model + the inference prep bundle. Falls back to CPU if GPU load fails."""
    from maia2 import model, inference
    try:
        m = model.from_pretrained(type=type, device=device)
    except Exception:
        m = model.from_pretrained(type=type, device="cpu")
    return m, inference.prepare()


class Maia2Bot(Player):
    def __init__(self, model, prep, self_elo: int, opp_elo: int | None = None, seed: int = 0):
        from maia2 import inference
        self._inference = inference
        self.model = model
        self.prep = prep
        self.self_elo = int(self_elo)
        self.opp_elo = int(opp_elo if opp_elo is not None else self_elo)
        self.rng = random.Random(seed)

    def choose_move(self, board: chess.Board) -> chess.Move:
        move_probs, _ = self._inference.inference_each(
            self.model, self.prep, board.fen(), self.self_elo, self.opp_elo)
        legal = {m.uci() for m in board.legal_moves}
        items = [(uci, p) for uci, p in move_probs.items() if uci in legal and p > 0]
        if not items:
            return self.rng.choice(list(board.legal_moves))
        ucis, weights = zip(*items)
        return chess.Move.from_uci(self.rng.choices(ucis, weights=weights, k=1)[0])
