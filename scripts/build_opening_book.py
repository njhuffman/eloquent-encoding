#!/usr/bin/env python3
"""Build per-elo-band opening books from a Lichess .pgn.zst (bucket by White elo)."""
from __future__ import annotations
import argparse, io
from pathlib import Path
import chess.pgn
from tqdm import tqdm
from dataset_generation.stream import iter_pgn_games_from_zstd_binary
from style_policy.opening_book import BookBuilder

_BANDS = [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]


def _white_elo(headers) -> int | None:
    raw = headers.get("WhiteElo")
    if raw is None or raw == "?":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def build(pgn_zst_path, out_dir, *, n_plies: int = 24, per_band_target: int = 100000,
          time_control: str = "600+0", min_support: float = 0.001) -> list[int]:
    bld = BookBuilder(n_plies=n_plies)
    counts = {b: 0 for b in _BANDS}
    with open(pgn_zst_path, "rb") as raw:
        for text in tqdm(iter_pgn_games_from_zstd_binary(raw), unit=" games"):
            if all(counts[b] >= per_band_target for b in _BANDS):
                break
            game = chess.pgn.read_game(io.StringIO(text))
            if game is None or game.headers.get("TimeControl") != time_control:
                continue
            we = _white_elo(game.headers)
            if we is None or not (1000 <= we <= 1999):
                continue
            band = (we // 100) * 100
            if counts[band] >= per_band_target:
                continue
            moves = list(game.mainline_moves())
            bld.add_game(we, moves)
            counts[band] += 1
    return bld.save_all(out_dir, min_support=min_support)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-plies", type=int, default=24)
    ap.add_argument("--per-band-target", type=int, default=100000)
    ap.add_argument("--min-support", type=float, default=0.001)
    args = ap.parse_args()
    bands = build(args.pgn, args.out, n_plies=args.n_plies,
                  per_band_target=args.per_band_target, min_support=args.min_support)
    print("wrote bands:", bands)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
