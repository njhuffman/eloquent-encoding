#!/usr/bin/env python3
"""
PGN → HDF5 for move predictor: frozen MAE embeddings, per-color padded history (last N
white-to-move and last N black-to-move snapshots), side_to_move, 3-way CE rows.
Stores promotion codes per slot for future use; training ignores them for now.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import deque
from pathlib import Path

import chess
import chess.pgn
import h5py
import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from embedding.board_encoding import board_to_tensor
from embedding.load import load_mae_by_name, load_mae_from_checkpoint
from move_predictor.config import (
    GAME_SKIP_PROB,
    HDF5_FLUSH_BATCH_SIZE,
    MOVE_SKIP_PROB as DEFAULT_MOVE_SKIP_PROB,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from move_predictor.encoding import move_to_from_to, promotion_code
from move_predictor.hdf5_io import ensure_move_h5, flush_move_h5, shuffle_columns


def _encoder_input_from_board(board: np.ndarray) -> np.ndarray:
    from embedding.config import BOARD_HEIGHT, BOARD_WIDTH

    mask = np.zeros((BOARD_HEIGHT, BOARD_WIDTH, 1), dtype=np.float32)
    return np.concatenate([board.astype(np.float32), mask], axis=-1)


def _encode_boards(
    model: torch.nn.Module,
    boards: list[np.ndarray],
    device: torch.device,
) -> np.ndarray:
    """boards: list of (8,8,18) -> (len, D) float32."""
    if not boards:
        return np.zeros((0, model.embedding_dim), dtype=np.float32)
    from embedding.config import BOARD_HEIGHT, BOARD_WIDTH

    enc_in = np.stack([_encoder_input_from_board(b) for b in boards], axis=0)
    with torch.no_grad():
        x = torch.from_numpy(enc_in).to(device)
        if x.shape[-1] == 19:
            x = x.permute(0, 3, 1, 2)
        emb = model.encoder(x)
        return emb.cpu().numpy().astype(np.float32)


def _assign_split(rng: random.Random) -> str:
    r = rng.random()
    if r < TRAIN_RATIO:
        return "train"
    if r < TRAIN_RATIO + VAL_RATIO:
        return "val"
    return "test"


def _sample_two_negatives(
    legal: list[chess.Move],
    chosen: chess.Move,
    rng: random.Random,
) -> tuple[chess.Move, chess.Move]:
    others = [m for m in legal if m != chosen]
    a, b = rng.sample(others, 2)
    return a, b


def _side_to_move_u8(b: chess.Board) -> int:
    return 0 if b.turn == chess.WHITE else 1


def _process_game(
    game: chess.pgn.Game,
    rng: random.Random,
    history_n: int,
    embedding_dim: int,
    move_skip_prob: float,
    model: torch.nn.Module,
    device: torch.device,
    buffers: dict[str, list],
) -> int:
    """Append sampled rows to buffers; returns count added."""
    board = game.board()
    prev_white: deque[chess.Board] = deque(maxlen=history_n)
    prev_black: deque[chess.Board] = deque(maxlen=history_n)
    added = 0
    for move in game.mainline_moves():
        if rng.random() < move_skip_prob:
            sk = board.copy()
            board.push(move)
            if sk.turn == chess.WHITE:
                prev_white.append(sk)
            else:
                prev_black.append(sk)
            continue

        legal = list(board.legal_moves)
        if len(legal) <= 2:
            sk = board.copy()
            board.push(move)
            if sk.turn == chess.WHITE:
                prev_white.append(sk)
            else:
                prev_black.append(sk)
            continue

        neg_a, neg_b = _sample_two_negatives(legal, move, rng)
        ff_ch, tt_ch = move_to_from_to(move)
        ff_a, tt_a = move_to_from_to(neg_a)
        ff_b, tt_b = move_to_from_to(neg_b)
        pr_ch, pr_a, pr_b = promotion_code(move), promotion_code(neg_a), promotion_code(neg_b)

        hw_boards = list(prev_white)
        hb_boards = list(prev_black)
        n_w, n_b = len(hw_boards), len(hb_boards)
        cur_tensor = board_to_tensor(board)

        cur_emb = _encode_boards(model, [cur_tensor], device)[0]
        hist_white_emb = np.zeros((history_n, embedding_dim), dtype=np.float32)
        if n_w:
            hist_white_emb[:n_w] = _encode_boards(
                model, [board_to_tensor(b) for b in hw_boards], device
            )
        hist_black_emb = np.zeros((history_n, embedding_dim), dtype=np.float32)
        if n_b:
            hist_black_emb[:n_b] = _encode_boards(
                model, [board_to_tensor(b) for b in hb_boards], device
            )

        sf, st, sp, label = shuffle_columns(
            [ff_ch, ff_a, ff_b],
            [tt_ch, tt_a, tt_b],
            [pr_ch, pr_a, pr_b],
            rng,
        )

        buffers["cur"].append(cur_emb)
        buffers["hist_w"].append(hist_white_emb)
        buffers["hist_b"].append(hist_black_emb)
        buffers["hlen_w"].append(n_w)
        buffers["hlen_b"].append(n_b)
        buffers["turn"].append(_side_to_move_u8(board))
        buffers["from"].append(np.asarray(sf, dtype=np.uint8))
        buffers["to"].append(np.asarray(st, dtype=np.uint8))
        buffers["prom"].append(np.asarray(sp, dtype=np.uint8))
        buffers["label"].append(label)
        buffers["fen"].append(board.fen())
        added += 1

        sk = board.copy()
        board.push(move)
        if sk.turn == chess.WHITE:
            prev_white.append(sk)
        else:
            prev_black.append(sk)

    return added


def main() -> int:
    parser = argparse.ArgumentParser(description="PGN → move predictor HDF5 (train/val/test).")
    parser.add_argument("pgn", type=Path, help="Input PGN")
    parser.add_argument("-o", "--output-dir", type=Path, default=None)
    parser.add_argument("--history-n", type=int, default=8)
    parser.add_argument("--game-skip-prob", type=float, default=GAME_SKIP_PROB)
    parser.add_argument("--move-skip-prob", type=float, default=DEFAULT_MOVE_SKIP_PROB)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--embedding-model", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--flush-size", type=int, default=HDF5_FLUSH_BATCH_SIZE)
    args = parser.parse_args()

    if not args.pgn.exists():
        print(f"Error: PGN not found: {args.pgn}", file=sys.stderr)
        return 1
    if args.embedding_model is None and args.checkpoint is None:
        print("Error: pass --embedding-model or --checkpoint", file=sys.stderr)
        return 1

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.checkpoint is not None:
        model = load_mae_from_checkpoint(args.checkpoint, device=device)
    else:
        model = load_mae_by_name(args.embedding_model, repo_root=_REPO_ROOT, device=device)
    model.eval()
    embedding_dim = int(model.embedding_dim)
    history_n = args.history_n

    out_dir = args.output_dir or args.pgn.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    train_path = out_dir / "train.h5"
    val_path = out_dir / "val.h5"
    test_path = out_dir / "test.h5"

    train_h5 = val_h5 = test_h5 = None
    train_next = val_next = test_next = 0
    totals = {"train": 0, "val": 0, "test": 0}

    def make_buffers() -> dict[str, list]:
        return {
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

    train_buf = make_buffers()
    val_buf = make_buffers()
    test_buf = make_buffers()

    def flush_split(name: str, buf: dict, h5_obj: h5py.File | None, next_idx: int) -> tuple[h5py.File | None, int]:
        if not buf["cur"]:
            return h5_obj, next_idx
        if h5_obj is None:
            path = {"train": train_path, "val": val_path, "test": test_path}[name]
            if path.exists():
                path.unlink()
            h5_obj = ensure_move_h5(path, embedding_dim, history_n)
            next_idx = 0
        next_idx = flush_move_h5(
            h5_obj,
            embedding_dim=embedding_dim,
            history_n=history_n,
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
        return h5_obj, next_idx

    game_count = 0
    with open(args.pgn, encoding="utf-8", errors="replace") as f:
        pbar = tqdm(desc="Games", unit=" games", dynamic_ncols=True, file=sys.stderr)
        while True:
            if args.max_samples is not None and sum(totals.values()) >= args.max_samples:
                break
            game = chess.pgn.read_game(f)
            if game is None:
                break
            game_count += 1
            if rng.random() < args.game_skip_prob:
                pbar.update(1)
                continue

            split = _assign_split(rng)
            buf = {"train": train_buf, "val": val_buf, "test": test_buf}[split]
            added = _process_game(
                game, rng, history_n, embedding_dim, args.move_skip_prob, model, device, buf
            )
            totals[split] += added
            pbar.update(1)
            pbar.set_postfix(**{k: totals[k] for k in totals}, refresh=False)

            if len(train_buf["cur"]) >= args.flush_size:
                train_h5, train_next = flush_split("train", train_buf, train_h5, train_next)
            if len(val_buf["cur"]) >= args.flush_size:
                val_h5, val_next = flush_split("val", val_buf, val_h5, val_next)
            if len(test_buf["cur"]) >= args.flush_size:
                test_h5, test_next = flush_split("test", test_buf, test_h5, test_next)

    pbar.close()

    train_h5, train_next = flush_split("train", train_buf, train_h5, train_next)
    val_h5, val_next = flush_split("val", val_buf, val_h5, val_next)
    test_h5, test_next = flush_split("test", test_buf, test_h5, test_next)

    for h in (train_h5, val_h5, test_h5):
        if h is not None:
            h.close()

    print(file=sys.stderr)
    print(f"Games seen: {game_count}", file=sys.stderr)
    for name, path in [("train", train_path), ("val", val_path), ("test", test_path)]:
        print(f"{name}: {totals[name]} samples -> {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
