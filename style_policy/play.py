"""Bot interface + arena for bot-vs-bot play.

`Player.choose_move(board) -> chess.Move` is the single integration seam — a UCI or
lichess-bot adapter is a thin wrapper over it. Everything uses python-chess types.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import torch
import chess
from style_policy.model import BasePolicy
from style_policy.model_spec import elo_to_bucket
from style_policy.legal_mask import u64_to_mask
from style_policy.board_encode import board_to_packed, legal_from_u64, legal_to_u64

_NEG = float("-inf")


class Player(ABC):
    @abstractmethod
    def choose_move(self, board: chess.Board) -> chess.Move: ...

    def reset(self) -> None:  # called at the start of each game; override if stateful
        pass


class PolicyBot(Player):
    """Plays by weighted sampling from the elo-conditioned move predictor:
    sample from-square ~ P(from | board, elo), then to-square ~ P(to | board, from, elo).
    temperature<1 sharpens toward the argmax (stronger/more deterministic), >1 flattens."""

    def __init__(self, checkpoint_path: str, elo: int, *, device: str = "cpu",
                 temperature: float = 1.0, seed: int | None = None):
        ck = torch.load(checkpoint_path, map_location=device)
        self.model = BasePolicy.from_config(ck["architecture"]).to(device)
        _loaded = self.model.load_state_dict(ck["model"], strict=False)
        assert not _loaded.unexpected_keys and all(k.startswith("value_head") for k in _loaded.missing_keys), \
            f"checkpoint mismatch: unexpected={_loaded.unexpected_keys} missing={_loaded.missing_keys}"
        self.model.eval()
        self.device = device
        self.n_elo = int(ck["architecture"]["n_elo_buckets"])
        self.elo = int(elo)
        self.temperature = float(temperature)
        self.gen = torch.Generator().manual_seed(seed if seed is not None else 0)
        self._elo_idx = elo_to_bucket(torch.tensor([self.elo]), self.n_elo).to(device)

    def _sample(self, logits: torch.Tensor, legal_u64: int) -> int:
        # np.uint64 -> int64 reinterpret (bit pattern preserved) avoids overflow when bit 63 (h8) is set
        u = torch.from_numpy(np.array([legal_u64], dtype=np.uint64)).to(torch.int64)
        mask = u64_to_mask(u).to(logits.device)
        logits = logits.masked_fill(~mask, _NEG) / self.temperature
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs[0], 1, generator=self.gen).item())

    @torch.no_grad()
    def choose_move(self, board: chess.Board) -> chess.Move:
        pk = torch.from_numpy(board_to_packed(board)[None]).to(self.device)
        _, squares = self.model.encode(pk)
        from_sq = self._sample(self.model.from_head(squares, elo_idx=self._elo_idx), legal_from_u64(board))
        to_logits = self.model.to_head(squares, torch.tensor([from_sq], device=self.device), elo_idx=self._elo_idx)
        to_sq = self._sample(to_logits, legal_to_u64(board, from_sq))
        mv = chess.Move(from_sq, to_sq)
        if mv not in board.legal_moves:                       # promotion: default to queen
            mv = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
        return mv


def play_game(white: Player, black: Player, *, max_plies: int = 400) -> str:
    """Play one game; return result string '1-0' / '0-1' / '1/2-1/2'."""
    board = chess.Board()
    white.reset(); black.reset()
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        mv = (white if board.turn == chess.WHITE else black).choose_move(board)
        board.push(mv)
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return "1/2-1/2"
    return "1-0" if outcome.winner == chess.WHITE else "0-1"


def play_match(white: Player, black: Player, n_games: int, *, max_plies: int = 400) -> dict:
    """n_games with fixed colors -> {white_wins, black_wins, draws}."""
    r = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    for _ in range(n_games):
        r[play_game(white, black, max_plies=max_plies)] += 1
    return {"white_wins": r["1-0"], "black_wins": r["0-1"], "draws": r["1/2-1/2"], "n": n_games}
