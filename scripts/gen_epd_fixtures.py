#!/usr/bin/env python3
"""Emit {epd, fen} parity cases (python-chess) for the TS opening-book key test."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import chess

_FENS = [
    chess.STARTING_FEN,
    "rnbqkbnr/ppp2ppp/4p3/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",   # legal e.p. (exd6)
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",     # spurious e.p. (no legal capture)
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",  # no e.p. midgame
    "r3k2r/8/8/8/8/8/8/R3K2R b Kq - 0 1",                              # partial castling
]


def build_epd_cases() -> list[dict]:
    # Normalize the fen through python-chess so the stored fen matches its epd convention,
    # then keep the ORIGINAL fen string too so chess.js loads the same position.
    out = []
    for fen in _FENS:
        b = chess.Board(fen)
        out.append({"fen": fen, "epd": b.epd()})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="web/src/inference/__fixtures__/epd_cases.json")
    args = ap.parse_args()
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(build_epd_cases(), indent=2))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
