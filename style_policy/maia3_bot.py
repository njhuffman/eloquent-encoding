"""Maia3 as a Player: drives the maia3-* UCI engine subprocess on CPU (never GPU — a training job
owns the only GPU). Argmax play at nodes=1 (Temperature default 0.0), conditioned on SelfElo/OppoElo
sent via UCI setoption. The engine reconstructs move history itself (--use-uci-history) from the
`position startpos moves ...` line, so we hand it the full game move stack each turn rather than a
bare FEN (falling back to FEN only when the board has no move_stack, e.g. built from a FEN directly)."""
from __future__ import annotations
import os
import subprocess
import chess
from style_policy.play import Player


class Maia3Bot(Player):
    def __init__(self, model: str = "maia3-23m", *, self_elo: int, oppo_elo: int,
                 nodes: int = 1, device: str = "cpu", seed: int | None = None):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        cmd = [model, "--device", device, "--no-use-amp", "--use-uci-history"]
        if seed is not None:
            cmd += ["--seed", str(seed)]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1, env=env,
        )
        self.nodes = int(nodes)
        self.self_elo = int(self_elo)
        self.oppo_elo = int(oppo_elo)
        self._send("uci"); self._await("uciok")
        self._send("isready"); self._await("readyok", timeout=120)
        self._send_elos()

    def _send(self, s: str) -> None:
        self.proc.stdin.write(s + "\n"); self.proc.stdin.flush()

    def _await(self, token: str, timeout: float | None = None) -> None:
        # timeout is advisory (mirrors maia1_bot's blocking iterate-until-found pattern); the first
        # isready after spawn triggers checkpoint load/download so callers pass a generous value.
        for line in self.proc.stdout:
            if line.strip().startswith(token):
                return
        raise RuntimeError(f"maia3 died awaiting {token!r}")

    def _send_elos(self) -> None:
        self._send(f"setoption name SelfElo value {self.self_elo}")
        self._send(f"setoption name OppoElo value {self.oppo_elo}")

    def set_elos(self, self_elo: int, oppo_elo: int) -> None:
        """Re-send SelfElo/OppoElo only if either changed since the last call/init."""
        self_elo, oppo_elo = int(self_elo), int(oppo_elo)
        if self_elo != self.self_elo or oppo_elo != self.oppo_elo:
            self.self_elo, self.oppo_elo = self_elo, oppo_elo
            self._send_elos()

    def choose_move(self, board: chess.Board) -> chess.Move | None:
        if board.move_stack:
            moves = " ".join(m.uci() for m in board.move_stack)
            self._send(f"position startpos moves {moves}")
        else:
            self._send(f"position fen {board.fen()}")
        self._send(f"go nodes {self.nodes}")
        for line in self.proc.stdout:
            if line.startswith("bestmove"):
                token = line.split()[1]
                if token == "(none)":
                    return None  # engine resigns / no legal move
                return chess.Move.from_uci(token)
        raise RuntimeError("maia3 produced no bestmove")

    def close(self) -> None:
        try:
            self._send("quit"); self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
