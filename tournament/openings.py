"""Sample balanced opening positions from a real PGN.zst -> openings.jsonl (generated ONCE, reused).
Take a mid-opening ply (default 10-16) from games that continue well past it, keep only near-material-
balanced positions so no bot starts winning. Each start is later played by both colors to cancel bias."""
from __future__ import annotations
import argparse, io, json, random
from pathlib import Path
import zstandard
import chess
import chess.pgn

_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def _material_diff(board: chess.Board) -> int:
    d = 0
    for sq, pc in board.piece_map().items():
        v = _VAL.get(pc.piece_type, 0)
        d += v if pc.color == chess.WHITE else -v
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="/mnt/eloquence_bulk/databases/lichess_db_standard_rated_2025-05_tc_600_0.pgn.zst")
    ap.add_argument("--out", default="tournament/openings.jsonl")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--ply-lo", type=int, default=10)
    ap.add_argument("--ply-hi", type=int, default=16)
    ap.add_argument("--max-material-diff", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    reader = zstandard.ZstdDecompressor().stream_reader(open(a.source, "rb"))
    text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    seen: set[str] = set()
    out = open(a.out, "w"); k = 0
    while k < a.n:
        game = chess.pgn.read_game(text)
        if game is None:
            break
        moves = list(game.mainline_moves())
        target = rng.randint(a.ply_lo, a.ply_hi)
        if len(moves) < target + 10:            # must continue well past the start
            continue
        board = game.board()
        for m in moves[:target]:
            board.push(m)
        if board.is_game_over() or abs(_material_diff(board)) > a.max_material_diff:
            continue
        fen = board.fen()
        if fen in seen:
            continue
        seen.add(fen)
        out.write(json.dumps({"opening_id": f"op{k:04d}", "fen": fen}) + "\n"); k += 1
    out.close()
    print(f"wrote {k} balanced openings (ply {a.ply_lo}-{a.ply_hi}, |material|<={a.max_material_diff}) -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
