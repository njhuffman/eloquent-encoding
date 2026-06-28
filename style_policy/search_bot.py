"""Expectimax search bot: policy proposes/opponent-models, WDL value head scores leaves.
Built on wdl_16M (the checkpoint with a trained value head)."""
from __future__ import annotations
import random
import numpy as np
import torch
import chess
from style_policy.model import BasePolicy
from style_policy.model_spec import elo_to_bucket
from style_policy.board_encode import board_to_packed, legal_from_u64, legal_to_u64
from style_policy.legal_mask import u64_to_mask
from style_policy.play import Player
from style_policy.search import expectimax

_NEG = float("-inf")


def _mask(u64: int) -> torch.Tensor:
    return u64_to_mask(torch.from_numpy(np.array([u64], dtype=np.uint64)).to(torch.int64))


def _escore(wdl_logits: torch.Tensor) -> float:
    p = torch.softmax(wdl_logits, dim=-1)[0]   # [loss, draw, win]
    return float(p[2] + 0.5 * p[1])


@torch.no_grad()
def policy_topk(model, board: chess.Board, elo_idx, k: int, device: str):
    """Top-k legal moves by joint P(from)*P(to|from), descending. One entry per distinct (from,to);
    a promotion (from,to) is represented by its queen move (the policy is from/to only)."""
    pk = torch.from_numpy(board_to_packed(board)[None]).to(device)
    _, squares = model.encode(pk)
    fl = model.from_head(squares, elo_idx=elo_idx)
    fprob = torch.softmax(fl.masked_fill(~_mask(legal_from_u64(board)).to(fl.device), _NEG), dim=-1)[0]
    to_cache: dict[int, torch.Tensor] = {}
    best: dict[tuple, tuple] = {}
    for mv in board.legal_moves:
        key = (mv.from_square, mv.to_square)
        if key in best:
            continue
        f = mv.from_square
        if f not in to_cache:
            tl = model.to_head(squares, torch.tensor([f], device=device), elo_idx=elo_idx)
            to_cache[f] = torch.softmax(tl.masked_fill(~_mask(legal_to_u64(board, f)).to(tl.device), _NEG), dim=-1)[0]
        p = float(fprob[f] * to_cache[f][mv.to_square])
        cm = mv if mv.promotion in (None, chess.QUEEN) else chess.Move(f, mv.to_square, promotion=chess.QUEEN)
        best[key] = (cm, p)
    return sorted(best.values(), key=lambda x: -x[1])[:k]


class ExpectimaxBot(Player):
    def __init__(self, checkpoint, elo, depth, *, width=4, device="cpu", seed=0,
                 opening_book=None, book_threshold=0.01):
        ck = torch.load(checkpoint, map_location=device)
        self.model = BasePolicy.from_config(ck["architecture"]).to(device)
        _loaded = self.model.load_state_dict(ck["model"], strict=False)
        assert not _loaded.unexpected_keys and all(k.startswith("value_head") for k in _loaded.missing_keys), \
            f"checkpoint mismatch: unexpected={_loaded.unexpected_keys} missing={_loaded.missing_keys}"
        self.model.eval()
        self.device = device
        self.depth = int(depth)
        self.width = int(width)
        n_elo = int(ck["architecture"]["n_elo_buckets"])
        self._elo_idx = elo_to_bucket(torch.tensor([int(elo)]), n_elo).to(device)
        self.opening_book = opening_book
        self.book_threshold = float(book_threshold)
        self._book_rng = random.Random(seed)

    @torch.no_grad()
    def choose_move(self, board: chess.Board) -> chess.Move:
        if self.opening_book is not None:
            mv = self.opening_book.lookup(board, self.book_threshold, self._book_rng)
            if mv is not None:
                return mv
        if self.depth == 0:
            return policy_topk(self.model, board, self._elo_idx, self.width, self.device)[0][0]
        bot_color = board.turn

        def expand(b):
            if b.is_game_over():
                return []
            out = []
            for mv, p in policy_topk(self.model, b, self._elo_idx, self.width, self.device):
                c = b.copy(stack=False); c.push(mv)
                out.append((mv, c, p))
            return out

        def is_max(b):
            return b.turn == bot_color

        def leaf_value(b):
            if b.is_game_over():
                o = b.outcome()
                if o is None or o.winner is None:
                    return 0.5
                return 1.0 if o.winner == bot_color else 0.0
            pk = torch.from_numpy(board_to_packed(b)[None]).to(self.device)
            e = _escore(self.model.value_head(self.model.encode(pk)[0], elo_idx=self._elo_idx))
            return e if b.turn == bot_color else 1.0 - e

        _, move = expectimax(board, self.depth, expand, leaf_value, is_max)
        return move
