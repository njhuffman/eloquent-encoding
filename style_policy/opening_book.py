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


class BookBuilder:
    def __init__(self, n_plies: int = 24):
        self.n_plies = int(n_plies)
        self._bands: dict[int, dict[str, dict]] = {}
        self._totals: dict[int, int] = {}

    def add_game(self, white_elo: int, moves: list[chess.Move]) -> None:
        if not (1000 <= int(white_elo) <= 1999):
            return
        band = (int(white_elo) // 100) * 100
        positions = self._bands.setdefault(band, {})
        self._totals[band] = self._totals.get(band, 0) + 1
        board = chess.Board()
        n_played = 0
        for i, mv in enumerate(moves):
            if i >= self.n_plies:
                break
            entry = positions.setdefault(board.epd(), {"n": 0, "moves": {}})
            entry["n"] += 1
            u = mv.uci()
            entry["moves"][u] = entry["moves"].get(u, 0) + 1
            board.push(mv)

    def finalize(self, min_support: float = 0.001) -> dict[int, OpeningBook]:
        out: dict[int, OpeningBook] = {}
        for band, positions in self._bands.items():
            total = self._totals[band]
            kept = {epd: e for epd, e in positions.items() if e["n"] / total >= min_support}
            out[band] = OpeningBook(total, kept)
        return out

    def save_all(self, out_dir, min_support: float = 0.001) -> list[int]:
        out_dir = Path(out_dir)
        written = []
        for band, book in self.finalize(min_support).items():
            if not book.positions:
                continue
            data = {"band": band, "total_games": book.total_games, "positions": book.positions}
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"band_{band}.json").write_text(json.dumps(data))
            written.append(band)
        return sorted(written)
