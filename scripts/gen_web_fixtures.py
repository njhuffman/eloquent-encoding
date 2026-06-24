#!/usr/bin/env python3
"""Generate JSON fixtures so the TS inference tests can check against PyTorch outputs."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch, chess
from style_policy.model import BasePolicy
from style_policy.model_spec import elo_to_bucket
from style_policy.board_encode import board_to_packed, legal_from_u64, legal_to_u64
from style_policy.packed_codec import packed_to_board_tensor

DEFAULT_FENS = [chess.STARTING_FEN,
                "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
                "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1",
                "rnbqkbnr/pp1ppppp/8/2pP4/8/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 3"]


def _bits(u64: int) -> list[bool]:
    return [bool((u64 >> i) & 1) for i in range(64)]


def build_cases(checkpoint_path: str, fens: list[str], elo: int) -> dict:
    ck = torch.load(checkpoint_path, map_location="cpu")
    policy = BasePolicy.from_config(ck["architecture"]); policy.load_state_dict(ck["model"], strict=False); policy.eval()  # old policy-only checkpoints predate the value head
    n = int(ck["architecture"]["n_elo_buckets"])
    bucket = int(elo_to_bucket(torch.tensor([elo]), n).item())
    cases = []
    for fen in fens:
        board = chess.Board(fen)
        bt = packed_to_board_tensor(board_to_packed(board)).float()
        with torch.no_grad():
            cls, sq = policy.encoder(bt)
            fl = policy.from_head(sq, elo_idx=torch.tensor([bucket]))[0]
            from_sq = int(fl.masked_fill(~torch.tensor(_bits(legal_from_u64(board))), float("-inf")).argmax())
            tl = policy.to_head(sq, torch.tensor([from_sq]), elo_idx=torch.tensor([bucket]))[0]
            to_legal = _bits(legal_to_u64(board, from_sq))
            to_sq = int(tl.masked_fill(~torch.tensor(to_legal), float("-inf")).argmax())
            value_logits = policy.value_head(cls, elo_idx=torch.tensor([bucket]))[0]
        cases.append({
            "fen": fen, "elo": elo, "bucket": bucket,
            "board_tensor": bt.reshape(-1).tolist(),
            "from_logits": fl.tolist(), "legal_from": _bits(legal_from_u64(board)),
            "to_from_sq": from_sq, "to_logits": tl.tolist(), "legal_to": to_legal,
            "value_logits": value_logits.tolist(),
            "top_move_uci": chess.Move(from_sq, to_sq).uci(),
        })
    return {"d_model": int(ck["architecture"]["d_model"]), "n_elo_buckets": n, "cases": cases}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="style_policy_checkpoints/base_64M/base_64M_stage_1.pt")
    ap.add_argument("--out", default="web/src/inference/__fixtures__/cases.json")
    ap.add_argument("--elo", type=int, default=1500)
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_cases(args.checkpoint, DEFAULT_FENS, args.elo)))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
