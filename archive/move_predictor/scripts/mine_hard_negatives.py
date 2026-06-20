#!/usr/bin/env python3
"""
Build stage-2 HDF5: chosen move, hardest non-chosen (per stage-1 model), random non-chosen.
Expects input HDF5 from pgn_to_move_hdf5 (includes fen + promotion columns).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import chess
import h5py
import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from move_predictor.config import HDF5_FLUSH_BATCH_SIZE
from move_predictor.encoding import move_to_from_to, promotion_code
from move_predictor.hdf5_io import ensure_move_h5, flush_move_h5, shuffle_columns
from move_predictor.model import MovePredictor


def _load_model(ckpt: Path, device: torch.device) -> MovePredictor:
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model = MovePredictor(
        embedding_dim=int(state["embedding_dim"]),
        history_n=int(state["history_n"]),
        move_emb_dim=int(state.get("move_emb_dim", 32)),
        turn_emb_dim=int(state.get("turn_emb_dim", 4)),
        gru_hidden=int(state.get("gru_hidden", 128)),
        gru_num_layers=int(state.get("gru_num_layers", 1)),
        mlp_hidden=int(state.get("mlp_hidden", 256)),
        dropout=float(state.get("dropout", 0.1)),
    )
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _chosen_move(from_sq: np.ndarray, to_sq: np.ndarray, prom: np.ndarray, label: int) -> chess.Move:
    return chess.Move(
        int(from_sq[label]),
        int(to_sq[label]),
        promotion=chess.PieceType(int(prom[label])) if int(prom[label]) != 0 else None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine hard negatives using stage-1 checkpoint")
    parser.add_argument("--input-h5", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--flush-size", type=int, default=HDF5_FLUSH_BATCH_SIZE)
    args = parser.parse_args()

    if not args.input_h5.is_file():
        print(f"Error: not found {args.input_h5}", file=sys.stderr)
        return 1

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rng = random.Random(args.seed)
    model = _load_model(args.checkpoint, device)

    if args.output.exists():
        args.output.unlink()

    h5_out = None
    next_idx = 0
    buf = {
        "cur": [],
        "hist_w": [],
        "hist_b": [],
        "hlen_w": [],
        "hlen_b": [],
        "turn": [],
        "from": [],
        "to": [],
        "prom": [],
        "label": [],
        "fen": [],
    }

    with h5py.File(args.input_h5, "r") as f_in:
        n = f_in["cur_emb"].shape[0]
        emb_d = int(f_in.attrs["embedding_dim"])
        hist_n = int(f_in.attrs["history_n"])

        for i in tqdm(range(n), desc="Mining", file=sys.stderr):
            fen = f_in["fen"][i]
            if isinstance(fen, bytes):
                fen = fen.decode("utf-8")
            board = chess.Board(fen)
            label = int(f_in["label"][i])
            ff = np.asarray(f_in["from_sq"][i], dtype=np.int64)
            tt = np.asarray(f_in["to_sq"][i], dtype=np.int64)
            pp = np.asarray(f_in["promotion"][i], dtype=np.int64)
            chosen = _chosen_move(ff, tt, pp, label)

            legal = list(board.legal_moves)
            others = [m for m in legal if m != chosen]
            if len(others) < 2:
                continue

            cur = torch.from_numpy(np.asarray(f_in["cur_emb"][i], dtype=np.float32)).unsqueeze(0).to(device)
            hw = torch.from_numpy(np.asarray(f_in["hist_white_emb"][i], dtype=np.float32)).unsqueeze(0).to(device)
            hb = torch.from_numpy(np.asarray(f_in["hist_black_emb"][i], dtype=np.float32)).unsqueeze(0).to(device)
            lw = torch.tensor([int(f_in["hist_white_len"][i])], dtype=torch.long, device=device)
            lb = torch.tensor([int(f_in["hist_black_len"][i])], dtype=torch.long, device=device)
            turn = torch.tensor([int(f_in["side_to_move"][i])], dtype=torch.long, device=device)

            ofs = torch.tensor([[m.from_square for m in others]], dtype=torch.long, device=device)
            ots = torch.tensor([[m.to_square for m in others]], dtype=torch.long, device=device)
            with torch.no_grad():
                scores = model.score_moves(cur, hw, hb, lw, lb, turn, ofs, ots)[0]
            hard_j = int(scores.argmax().item())
            easy_pool = [j for j in range(len(others)) if j != hard_j]
            easy_j = rng.choice(easy_pool)
            hard_m, easy_m = others[hard_j], others[easy_j]

            ff_ch, tt_ch = move_to_from_to(chosen)
            pr_ch = promotion_code(chosen)
            ff_h, tt_h = move_to_from_to(hard_m)
            pr_h = promotion_code(hard_m)
            ff_e, tt_e = move_to_from_to(easy_m)
            pr_e = promotion_code(easy_m)

            sf, st, sp, new_label = shuffle_columns(
                [ff_ch, ff_h, ff_e],
                [tt_ch, tt_h, tt_e],
                [pr_ch, pr_h, pr_e],
                rng,
            )

            buf["cur"].append(np.asarray(f_in["cur_emb"][i], dtype=np.float32))
            buf["hist_w"].append(np.asarray(f_in["hist_white_emb"][i], dtype=np.float32))
            buf["hist_b"].append(np.asarray(f_in["hist_black_emb"][i], dtype=np.float32))
            buf["hlen_w"].append(int(f_in["hist_white_len"][i]))
            buf["hlen_b"].append(int(f_in["hist_black_len"][i]))
            buf["turn"].append(int(f_in["side_to_move"][i]))
            buf["from"].append(np.asarray(sf, dtype=np.uint8))
            buf["to"].append(np.asarray(st, dtype=np.uint8))
            buf["prom"].append(np.asarray(sp, dtype=np.uint8))
            buf["label"].append(new_label)
            buf["fen"].append(fen)

            if len(buf["cur"]) >= args.flush_size:
                if h5_out is None:
                    h5_out = ensure_move_h5(args.output, emb_d, hist_n)
                    next_idx = 0
                next_idx = flush_move_h5(
                    h5_out,
                    embedding_dim=emb_d,
                    history_n=hist_n,
                    cur_list=buf["cur"],
                    hist_white_list=buf["hist_w"],
                    hist_black_list=buf["hist_b"],
                    hlen_white_list=buf["hlen_w"],
                    hlen_black_list=buf["hlen_b"],
                    side_to_move_list=buf["turn"],
                    from_list=buf["from"],
                    to_list=buf["to"],
                    prom_list=buf["prom"],
                    label_list=buf["label"],
                    fen_list=buf["fen"],
                    next_idx=next_idx,
                )
                for k in buf:
                    buf[k].clear()

        if buf["cur"]:
            if h5_out is None:
                h5_out = ensure_move_h5(args.output, emb_d, hist_n)
                next_idx = 0
            flush_move_h5(
                h5_out,
                embedding_dim=emb_d,
                history_n=hist_n,
                cur_list=buf["cur"],
                hist_white_list=buf["hist_w"],
                hist_black_list=buf["hist_b"],
                hlen_white_list=buf["hlen_w"],
                hlen_black_list=buf["hlen_b"],
                side_to_move_list=buf["turn"],
                from_list=buf["from"],
                to_list=buf["to"],
                prom_list=buf["prom"],
                label_list=buf["label"],
                fen_list=buf["fen"],
                next_idx=next_idx,
            )

    if h5_out is not None:
        h5_out.close()

    print(f"Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
