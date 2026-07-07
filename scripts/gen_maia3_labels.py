"""Phase B: build a distillation label set. Stream a training-month PGN, extract candidate
positions in MY packed schema (drop-in for PackedMoveDataset), and for each attach Maia-3's
soft policy as two factored targets:
  maia_from[64] = Maia-3 P(from-square)              (full from-marginal)
  maia_to[64]   = Maia-3 P(to-square | true human from)   (to-dist conditioned on the played from)
One h5 -> trains identically under CE-on-human OR factored-KL-to-Maia-3 (only the loss differs).
Maia-3 runs BATCHED on GPU. Per-band capped, balanced.
"""
from __future__ import annotations
import argparse, io, sys, os, time, numpy as np, torch, h5py, zstandard
from collections import deque, defaultdict
import chess, chess.pgn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_generation.candidate_collect import collect_candidate_positions, board_at_ply
from style_policy.board_encode import board_to_packed, legal_from_u64, legal_to_u64
from style_policy.packed_codec import PACKED_BOARD_LEN
from maia3.uci import parse_args as m3_parse_args, Maia3UCIEngine
from maia3.dataset import tokenize_board, get_historical_tokens, get_legal_moves_mask

BANDS = list(range(1000, 2200, 100))
def band_of(elo): return None if (elo < 1000 or elo >= 2200) else int(min(2100, max(1000, (elo // 100) * 100)))
_NEG = float("-inf")


def build_engine(model_name, device):
    cfg = m3_parse_args(["--model", model_name, "--device", device, "--use-uci-history"])
    eng = Maia3UCIEngine(cfg); eng.ensure_model_loaded()
    return eng


def m3_tokens(eng, toks_list, ply):
    hist = deque(toks_list[max(0, ply - eng.cfg.history + 1): ply + 1], maxlen=eng.cfg.history)
    return get_historical_tokens(hist, eng.cfg, base=0.0, inc=0.0, clk_left_before=0.0, clk_ponder=0.0)


class Writer:
    def __init__(self, path, chunk=8192):
        self.f = h5py.File(path, "w"); self.n = 0; self.buf = defaultdict(list)
        specs = {
            "packed_pre": (np.uint8, (PACKED_BOARD_LEN,)), "from_sq": (np.uint8, ()), "to_sq": (np.uint8, ()),
            "promotion": (np.uint8, ()), "from_legal_u64": (np.uint64, ()), "to_legal_u64": (np.uint64, ()),
            "elo_to_move": (np.int16, ()), "opp_elo": (np.int16, ()), "result": (np.int8, ()),
            "hist_from": (np.int8, (4,)), "hist_to": (np.int8, (4,)), "hist_cap": (np.int8, (4,)),
            "maia_from": (np.float16, (64,)), "maia_to": (np.float16, (64,)),
        }
        self.specs = specs
        for k, (dt, sh) in specs.items():
            self.f.create_dataset(k, shape=(0, *sh), maxshape=(None, *sh), dtype=dt,
                                  chunks=(chunk, *sh), compression=None)

    def add(self, row):
        for k in self.specs: self.buf[k].append(row[k])
        self.n += 1
        if len(self.buf["from_sq"]) >= 50000: self.flush()

    def flush(self):
        if not self.buf["from_sq"]: return
        m = len(self.buf["from_sq"])
        for k, (dt, sh) in self.specs.items():
            d = self.f[k]; d.resize(d.shape[0] + m, axis=0)
            d[-m:] = np.asarray(self.buf[k], dtype=dt)
        self.buf = defaultdict(list)

    def close(self): self.flush(); self.f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", default="/mnt/eloquence_bulk/databases/lichess_db_standard_rated_2025-01_tc_600_0.pgn.zst")
    ap.add_argument("--out", default="/mnt/eloquence_bulk/databases/maia3_distill_labels.h5")
    ap.add_argument("--maia3-model", default="maia3-23m")
    ap.add_argument("--per-band", type=int, default=1_333_000)
    ap.add_argument("--skip-opening", type=int, default=4)
    ap.add_argument("--sample-prob", type=float, default=0.5)
    ap.add_argument("--max-per-game", type=int, default=4)
    ap.add_argument("--max-games", type=int, default=0)  # 0 = unlimited
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); dev = a.device
    rng = np.random.default_rng(1)
    eng = build_engine(a.maia3_model, dev); print("maia3 loaded", flush=True)
    W = Writer(a.out)
    need = {b: a.per_band for b in range(len(BANDS))}
    buf = []  # (tokens_tensor, mask_tensor, board, self_elo, opp_elo, myrow)
    t0 = time.time(); scanned = 0; kept = 0

    def flush_batch():
        nonlocal kept
        if not buf: return
        toks = torch.stack([b[0] for b in buf]).to(dev)
        se = torch.tensor([b[3] for b in buf], dtype=torch.long, device=dev)
        oe = torch.tensor([b[4] for b in buf], dtype=torch.long, device=dev)
        with torch.no_grad():
            logits_move, _v, _ = eng.model(toks, se, oe)
        logits_move = logits_move.float()
        for i, (tk, mask, board, s_elo, o_elo, myrow) in enumerate(buf):
            lg = logits_move[i].masked_fill(~mask.to(dev), _NEG)
            probs = torch.softmax(lg, dim=-1)
            eng.board = board
            mf = np.zeros(64, np.float32); mt = np.zeros(64, np.float32)
            tf = int(myrow["from_sq"])
            for idx in mask.nonzero().flatten().tolist():
                mv = eng._move_from_index(idx)
                if mv is None: continue
                p = float(probs[idx]); mf[mv.from_square] += p
                if mv.from_square == tf: mt[mv.to_square] += p
            s = mt.sum(); mt = mt / s if s > 1e-9 else mt  # P(to | true from)
            myrow["maia_from"] = mf; myrow["maia_to"] = mt
            W.add(myrow); kept += 1
        buf.clear()

    dctx = zstandard.ZstdDecompressor()
    with open(a.pgn, "rb") as raw:
        text = io.TextIOWrapper(dctx.stream_reader(raw), encoding="utf-8", errors="replace")
        while any(v > 0 for v in need.values()):
            game = chess.pgn.read_game(text)
            if game is None: break
            scanned += 1
            if a.max_games and scanned > a.max_games: break
            if scanned % 20000 == 0:
                print(f"  games {scanned:,} | kept {kept:,} | remaining {sum(need.values()):,} | {kept/max(1,time.time()-t0):.0f} rows/s", flush=True)
            try:
                mainline, rows = collect_candidate_positions(game, skip_opening_plies=a.skip_opening, exclude_single_legal_move=True)
            except Exception:
                continue
            if not rows: continue
            # replay once for maia3 per-ply tokens
            b0 = chess.Board(); toks_list = [tokenize_board(b0)]
            for mv in mainline:
                b0.push(mv); toks_list.append(tokenize_board(b0))
            taken = 0
            for (ply, stm, elo_tm, opp_elo, result, mv, hist) in rows:
                bd = band_of(elo_tm)
                if bd is None: continue
                bi = (bd - 1000) // 100
                if need[bi] <= 0 or taken >= a.max_per_game: continue
                if rng.random() >= a.sample_prob: continue
                board = board_at_ply(mainline, ply)
                myrow = {
                    "packed_pre": board_to_packed(board),
                    "from_sq": mv.from_square, "to_sq": mv.to_square,
                    "promotion": (mv.promotion or 0),
                    "from_legal_u64": legal_from_u64(board) & 0xFFFFFFFFFFFFFFFF,
                    "to_legal_u64": legal_to_u64(board, mv.from_square) & 0xFFFFFFFFFFFFFFFF,
                    "elo_to_move": elo_tm, "opp_elo": opp_elo, "result": result,
                    "hist_from": [h[0] for h in hist], "hist_to": [h[1] for h in hist], "hist_cap": [h[2] for h in hist],
                }
                mask = get_legal_moves_mask(board, eng.all_moves_dict)
                buf.append((m3_tokens(eng, toks_list, ply), mask, board.copy(stack=False),
                            int(elo_tm), int(opp_elo), myrow))
                need[bi] -= 1; taken += 1
                if len(buf) >= a.batch: flush_batch()
    flush_batch(); W.close()
    print(f"\nDONE: {kept:,} rows in {(time.time()-t0)/60:.1f} min (scanned {scanned:,} games)")
    print(f"per-band kept: {[a.per_band - need[b] for b in range(len(BANDS))]}")


if __name__ == "__main__":
    main()
