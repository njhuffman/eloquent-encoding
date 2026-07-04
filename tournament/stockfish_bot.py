"""StockfishBot: the diverse, cleanly-separated reference opponent for the tournament. Wraps
python-chess's UCI engine; graded by Skill Level (0-20) and/or UCI_Elo, capped by depth for speed.
Skill<20 injects deliberate weakening (also gives game variety). Structurally different from the
human-imitation bots, so it doesn't compress against them."""
from __future__ import annotations
import chess
import chess.engine
from style_policy.play import Player

_SF = "/usr/games/stockfish"


class StockfishBot(Player):
    def __init__(self, *, sf_path: str = _SF, skill_level: int | None = None, elo: int | None = None,
                 depth: int = 8, movetime: float | None = None, threads: int = 1, seed: int = 0):
        self.engine = chess.engine.SimpleEngine.popen_uci(sf_path)
        opts: dict = {"Threads": int(threads)}
        if skill_level is not None:
            opts["Skill Level"] = int(skill_level)
        if elo is not None:
            opts["UCI_LimitStrength"] = True
            opts["UCI_Elo"] = int(elo)
        self.engine.configure(opts)
        self.limit = chess.engine.Limit(time=movetime) if movetime is not None else chess.engine.Limit(depth=int(depth))

    def choose_move(self, board: chess.Board) -> chess.Move:
        return self.engine.play(board, self.limit).move

    def close(self) -> None:
        try:
            self.engine.quit()
        except Exception:
            pass
