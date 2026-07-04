"""MultiBandBot: plays with the joint MultiBandPolicy checkpoint, routed to a chosen elo band,
feeding the last-n-ply move history during play (so a history-trained model plays at its real
strength, not K=0). Slots into the rate_bot / Maia2 harness like PolicyBot / BandHeadBot."""
from __future__ import annotations
import random
import numpy as np
import torch
import chess
from style_policy.play import Player
from style_policy.multiband_policy import MultiBandPolicy
from style_policy.board_encode import board_to_packed, legal_from_u64, legal_to_u64
from style_policy.legal_mask import u64_to_mask

_NEG = float("-inf")


def board_history(board: chess.Board, n_ply: int) -> tuple[list[int], list[int], list[int]]:
    """Last `n_ply` plies from the board, newest-first, as (from, to, cap) padded with absent
    (-1,-1,0). cap = 0 none / python-chess piece_type 1-5 / en passant -> 1 (pawn). Mirrors the
    mining history logic (captured piece read in the position BEFORE the move)."""
    hf = [-1] * n_ply; ht = [-1] * n_ply; hc = [0] * n_ply
    b = board.copy()
    for i in range(n_ply):
        if not b.move_stack:
            break
        m = b.pop()  # undo; b is now the position BEFORE m
        if b.is_en_passant(m):
            cap = 1
        elif b.is_capture(m):
            pc = b.piece_at(m.to_square)
            cap = pc.piece_type if pc is not None else 0
        else:
            cap = 0
        hf[i] = m.from_square; ht[i] = m.to_square; hc[i] = cap
    return hf, ht, hc


class MultiBandBot(Player):
    def __init__(self, checkpoint, elo: int, *, device: str = "cpu", temperature: float = 1.0,
                 seed=None, opening_book=None, book_threshold: float = 0.01, model=None, arch=None):
        if model is not None:               # share a pre-loaded model (avoids N GPU copies)
            self.model = model; self.arch = arch
        else:
            ck = torch.load(checkpoint, map_location=device)
            self.model = MultiBandPolicy.from_config(ck["architecture"]).to(device)
            self.model.load_state_dict(ck["model"]); self.model.eval()
            self.arch = ck["architecture"]
        self.n_ply = int(self.arch.get("n_history_ply", 0)) if self.arch.get("use_last_move") else 0
        self.g = int(self.model.head_index(torch.tensor([int(elo)])).item())  # routed band head
        self.elo = int(elo)
        self.device = device
        self.temperature = float(temperature)
        self.gen = torch.Generator(device=device).manual_seed(seed if seed is not None else 0)
        self.opening_book = opening_book
        self.book_threshold = float(book_threshold)
        self._book_rng = random.Random(seed if seed is not None else 0)

    def _sample(self, logits: torch.Tensor, legal_u64: int) -> int:
        mask = u64_to_mask(torch.from_numpy(np.array([legal_u64], dtype=np.uint64)).to(torch.int64)).to(self.device)
        logits = logits.masked_fill(~mask, _NEG) / self.temperature
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs[0], 1, generator=self.gen).item())

    @torch.no_grad()
    def choose_move(self, board: chess.Board) -> chess.Move:
        if self.opening_book is not None:
            mv = self.opening_book.lookup(board, self.book_threshold, self._book_rng)
            if mv is not None:
                return mv
        pk = torch.from_numpy(board_to_packed(board)[None]).to(self.device)
        hist = None
        if self.n_ply:
            hf, ht, hc = board_history(board, self.n_ply)
            hist = (torch.tensor([hf], device=self.device), torch.tensor([ht], device=self.device),
                    torch.tensor([hc], device=self.device))
        cls, squares = self.model.encode(pk, hist=hist)
        head = self.model.heads[self.g]
        from_sq = self._sample(head.from_logits(squares, cls), legal_from_u64(board))
        to_logits = head.to_logits(squares, torch.tensor([from_sq], device=self.device), cls)
        to_sq = self._sample(to_logits, legal_to_u64(board, from_sq))
        mv = chess.Move(from_sq, to_sq)
        if mv not in board.legal_moves:
            mv = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
        return mv
