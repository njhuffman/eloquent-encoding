#!/usr/bin/env python3
"""
Phase-2 gating: extract per-game, per-side style features from a streamed .pgn.zst,
using SAN move strings only (regex; no python-chess board replay -> fast).

Emits JSONL, one row per game:
  white_id, black_id, white_elo, black_elo, result (1/0.5/0 for White), plies,
  and w_<feat> / b_<feat> for each side. Aggregate downstream to (player,color) profiles.

Rate features are over moves AFTER `skip_plies` (skip opening book); `castled` is tracked
over the whole game. Only games matching --time-control with both elos are kept.
"""
from __future__ import annotations
import argparse, io, json, re, sys
from pathlib import Path
import zstandard
from tqdm import tqdm

_WHITE_RE = re.compile(r'\[White\s+"([^"]*)"\]')
_BLACK_RE = re.compile(r'\[Black\s+"([^"]*)"\]')
_WELO_RE = re.compile(r'\[WhiteElo\s+"(\d+)"\]')
_BELO_RE = re.compile(r'\[BlackElo\s+"(\d+)"\]')
_TC_RE = re.compile(r'\[TimeControl\s+"([^"]*)"\]')
_RES_RE = re.compile(r'\[Result\s+"([^"]*)"\]')
_COMMENT_RE = re.compile(r'\{[^}]*\}')
_MOVENUM_RE = re.compile(r'\d+\.(\.\.)?')
_SQUARE_RE = re.compile(r'[a-h][1-8]')
_RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}
_RESULT_TO_W = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


def _side_features(sans: list[str], skip_plies_side: int) -> dict:
    """sans = this side's moves in order. skip_plies_side = how many of this side's early moves to skip for rates."""
    castled = 0
    castle_move = -1
    for i, s in enumerate(sans):
        if s.startswith("O-O"):
            castled = 1
            castle_move = i
            break
    moves = sans[skip_plies_side:]
    n = len(moves)
    if n == 0:
        return None
    cap = chk = promo = pawn = minor = major = queen = king = opp_half = 0
    adv_sum = 0
    for s in moves:
        if "x" in s:
            cap += 1
        if s.endswith("+") or s.endswith("#"):
            chk += 1
        if "=" in s:
            promo += 1
        if s.startswith("O-O"):
            king += 1  # castling counts as a king move
        else:
            c = s[0]
            if c == "N" or c == "B":
                minor += 1
            elif c == "R" or c == "Q":
                major += 1
                if c == "Q":
                    queen += 1
            elif c == "K":
                king += 1
            else:
                pawn += 1
        m = _SQUARE_RE.findall(s)
        if m:
            adv_sum += int(m[-1][1])  # destination rank 1..8 (White perspective)
    return {"n": n, "capture_rate": cap / n, "check_rate": chk / n, "promo_rate": promo / n,
            "pawn_rate": pawn / n, "minor_rate": minor / n, "major_rate": major / n,
            "queen_rate": queen / n, "king_rate": king / n, "castled": castled,
            "castle_move": castle_move, "adv_sum": adv_sum}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn_zst", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-games", type=int, default=2_000_000)
    ap.add_argument("--time-control", default="600+0")
    ap.add_argument("--skip-plies", type=int, default=6, help="opening plies to skip for rate features")
    ap.add_argument("--min-plies", type=int, default=20, help="drop ultra-short games")
    args = ap.parse_args()

    skip_side = args.skip_plies // 2
    kept = 0
    with open(args.pgn_zst, "rb") as raw, open(args.out, "w") as out:
        reader = zstandard.ZstdDecompressor().stream_reader(raw)
        text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        hdr = {}
        movetext_lines: list[str] = []
        in_moves = False

        def flush(hdr, movetext):
            nonlocal kept
            if hdr.get("tc") != args.time_control:
                return
            if not (hdr.get("welo") and hdr.get("belo")):
                return
            mt = _COMMENT_RE.sub(" ", movetext)
            mt = _MOVENUM_RE.sub(" ", mt)
            toks = [t for t in mt.split() if t not in _RESULT_TOKENS and not t.startswith("$")]
            if len(toks) < args.min_plies:
                return
            w_sans = toks[0::2]
            b_sans = toks[1::2]
            wf = _side_features(w_sans, skip_side)
            bf = _side_features(b_sans, skip_side)
            if wf is None or bf is None:
                return
            # adv: mean destination rank, mapped to "forward" per side (White: rank; Black: 9-rank)
            wf["mean_adv"] = wf.pop("adv_sum") / wf["n"]
            bf["mean_adv"] = 9 - (bf.pop("adv_sum") / bf["n"])
            row = {"white_id": hdr["white"], "black_id": hdr["black"],
                   "white_elo": hdr["welo"], "black_elo": hdr["belo"],
                   "result": _RESULT_TO_W.get(hdr.get("result"), None), "plies": len(toks)}
            for k, v in wf.items():
                row["w_" + k] = v
            for k, v in bf.items():
                row["b_" + k] = v
            out.write(json.dumps(row) + "\n")
            kept += 1

        pbar = tqdm(text, unit=" lines", file=sys.stderr, mininterval=0.5)
        for line in pbar:
            s = line.strip()
            if s.startswith("[Event "):
                if hdr.get("white") is not None and movetext_lines:
                    flush(hdr, " ".join(movetext_lines))
                    if kept >= args.max_games:
                        break
                hdr = {}
                movetext_lines = []
                in_moves = False
                continue
            if s.startswith("["):
                for key, rgx in (("white", _WHITE_RE), ("black", _BLACK_RE), ("welo", _WELO_RE),
                                 ("belo", _BELO_RE), ("tc", _TC_RE), ("result", _RES_RE)):
                    m = rgx.match(s)
                    if m:
                        v = m.group(1)
                        hdr[key] = int(v) if key in ("welo", "belo") else v
                continue
            if s == "":
                in_moves = True
                continue
            if in_moves:
                movetext_lines.append(s)
            pbar.set_postfix(kept=kept, refresh=False)
        if hdr.get("white") is not None and movetext_lines and kept < args.max_games:
            flush(hdr, " ".join(movetext_lines))

    print(f"wrote {kept} games -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
