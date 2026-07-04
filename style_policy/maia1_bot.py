"""Maia1 as a Player: drives an lc0 UCI subprocess with a Maia1 weight file at nodes=1 (Maia1's
intended single-eval play). Policy temperature controls sampling (T=1.0 = sample the model's move
distribution, matching how we ran Maia2Bot). Each Maia1 rating is a SEPARATE network — the test is
whether that separates playing strength where Maia2's unified conditioning did not."""
from __future__ import annotations
import subprocess
import chess
from style_policy.play import Player


class Maia1Bot(Player):
    def __init__(self, weights: str, *, lc0: str = "lc0", temperature: float = 1.0,
                 nodes: int = 1, seed: int = 0):
        self.proc = subprocess.Popen(
            [lc0, f"--weights={weights}", "--backend=eigen",
             f"--temperature={temperature}"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
        )
        self.nodes = int(nodes)
        self._send("uci"); self._await("uciok")
        self._send("isready"); self._await("readyok")

    def _send(self, s: str) -> None:
        self.proc.stdin.write(s + "\n"); self.proc.stdin.flush()

    def _await(self, token: str) -> None:
        for line in self.proc.stdout:
            if line.strip().startswith(token):
                return
        raise RuntimeError(f"lc0 died awaiting {token!r}")

    def choose_move(self, board: chess.Board) -> chess.Move:
        self._send(f"position fen {board.fen()}")
        self._send(f"go nodes {self.nodes}")
        for line in self.proc.stdout:
            if line.startswith("bestmove"):
                return chess.Move.from_uci(line.split()[1])
        raise RuntimeError("lc0 produced no bestmove")

    def close(self) -> None:
        try:
            self._send("quit"); self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
