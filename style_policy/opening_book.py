"""Per-elo-band statistical opening book (PGN-derived). Play-side OpeningBook +
build-side BookBuilder. Positions keyed by board.epd() (transposition-merged)."""
from __future__ import annotations
import json
import random
from pathlib import Path
import chess


def elo_to_band(elo: int) -> int:
    """Lower bound of the 100-wide band, clamped to the built range [1000, 1900]."""
    return max(1000, min(1900, (int(elo) // 100) * 100))


class OpeningBook:
    def __init__(self, total_games: int, positions: dict[str, dict]):
        self.total_games = int(total_games)
        self.positions = positions  # {epd: {"n": int, "moves": {uci: int}}}

    def lookup(self, board: chess.Board, threshold: float, rand: random.Random) -> chess.Move | None:
        entry = self.positions.get(board.epd())
        if entry is None or self.total_games <= 0:
            return None
        if entry["n"] / self.total_games < threshold:
            return None
        legal = {m.uci(): m for m in board.legal_moves}
        items = [(legal[u], c) for u, c in entry["moves"].items() if u in legal]
        if not items:
            return None
        total = sum(c for _, c in items)
        r = rand.random() * total
        acc = 0.0
        for mv, c in items:
            acc += c
            if r <= acc:
                return mv
        return items[-1][0]

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"total_games": self.total_games, "positions": self.positions}))

    @classmethod
    def load(cls, path) -> "OpeningBook":
        d = json.loads(Path(path).read_text())
        return cls(d["total_games"], d["positions"])

    @classmethod
    def for_elo(cls, book_dir, elo: int) -> "OpeningBook | None":
        p = Path(book_dir) / f"band_{elo_to_band(elo)}.json"
        return cls.load(p) if p.exists() else None
