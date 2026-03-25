#!/usr/bin/env python3
"""
Convert PGN to HDF5 train/val/test with transition tuples for Chess-JEPA:
board_t, board after played move, K negative next boards, mover Elo.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import chess
import chess.pgn
import h5py
import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from embedding.board_encoding import board_to_tensor
from jepa.config import (
    BOARD_CHANNELS,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    FIRST_10_MOVES_SKIP_PROB,
    HDF5_FLUSH_BATCH_SIZE,
    NUM_NEGATIVES_DEFAULT,
    OPENING_SKIP_HALF_MOVES,
    REMAINING_MOVES_SKIP_PROB,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)


def _parse_elo(headers: chess.pgn.Headers, color: str) -> float:
    key = "WhiteElo" if color == "white" else "BlackElo"
    try:
        v = int(headers.get(key, 0) or 0)
        return float(v)
    except (ValueError, TypeError):
        return 0.0


def _sample_negatives(
    board_before: chess.Board,
    played: chess.Move,
    k: int,
    rng: random.Random,
) -> list[chess.Move]:
    legal = [m for m in board_before.legal_moves if m != played]
    if not legal:
        return []
    k = min(k, len(legal))
    return rng.sample(legal, k)


def _process_game(
    game: chess.pgn.Game,
    rng: random.Random,
    num_negatives: int,
    train_bufs: dict,
    val_bufs: dict,
    test_bufs: dict,
    split: str,
) -> tuple[int, int, int]:
    """Returns (train_added, val_added, test_added)."""
    board = game.board()
    headers = game.headers
    train_added = val_added = test_added = 0
    half_move = 0

    for move in game.mainline_moves():
        board_before = board.copy()
        side = board_before.turn
        elo_mover = _parse_elo(headers, "white" if side == chess.WHITE else "black")

        if (
            board_before.is_checkmate()
            or board_before.is_stalemate()
            or len(list(board_before.legal_moves)) <= 1
        ):
            board.push(move)
            half_move += 1
            continue

        neg_moves = _sample_negatives(board_before, move, num_negatives, rng)
        board.push(move)
        half_move += 1

        if half_move <= OPENING_SKIP_HALF_MOVES:
            continue
        if half_move <= OPENING_SKIP_HALF_MOVES + 10:
            if rng.random() < FIRST_10_MOVES_SKIP_PROB:
                continue
        else:
            if rng.random() < REMAINING_MOVES_SKIP_PROB:
                continue

        if len(neg_moves) < num_negatives:
            continue

        tensor_t = board_to_tensor(board_before)
        tensor_pos = board_to_tensor(board)
        negs = []
        for nm in neg_moves:
            bc = board_before.copy()
            bc.push(nm)
            negs.append(board_to_tensor(bc))
        neg_stack = np.stack(negs, axis=0).astype(np.float32, copy=False)

        row = {
            "board_t": tensor_t,
            "board_t_plus_1_pos": tensor_pos,
            "board_t_plus_1_negs": neg_stack,
            "elo": np.float32(elo_mover),
        }

        if split == "train":
            for key in train_bufs:
                train_bufs[key].append(row[key])
            train_added += 1
        elif split == "val":
            for key in val_bufs:
                val_bufs[key].append(row[key])
            val_added += 1
        else:
            for key in test_bufs:
                test_bufs[key].append(row[key])
            test_added += 1

    return train_added, val_added, test_added


def _assign_split(rng: random.Random) -> str:
    r = rng.random()
    if r < TRAIN_RATIO:
        return "train"
    if r < TRAIN_RATIO + VAL_RATIO:
        return "val"
    return "test"


def _flush_jepa_batch(
    h5: h5py.File,
    bufs: dict[str, list],
    dsets: dict[str, h5py.Dataset],
    next_idx: int,
    k_neg: int,
) -> int:
    if not bufs["board_t"]:
        return next_idx
    n = len(bufs["board_t"])
    new_size = next_idx + n
    for name in ("board_t", "board_t_plus_1_pos", "elo"):
        dsets[name].resize(new_size, axis=0)
        batch = np.stack(bufs[name], axis=0).astype(np.float32, copy=False)
        dsets[name][next_idx:new_size] = batch
        bufs[name].clear()
    dsets["board_t_plus_1_negs"].resize(new_size, axis=0)
    neg_batch = np.stack(bufs["board_t_plus_1_negs"], axis=0)
    dsets["board_t_plus_1_negs"][next_idx:new_size] = neg_batch
    bufs["board_t_plus_1_negs"].clear()
    assert int(h5.attrs.get("num_negatives_k", k_neg)) == k_neg
    return new_size


def _create_file(path: Path, k_neg: int) -> tuple[h5py.File, dict[str, h5py.Dataset]]:
    f = h5py.File(path, "w")
    f.attrs["num_negatives_k"] = k_neg
    ch = (1, BOARD_HEIGHT, BOARD_WIDTH, BOARD_CHANNELS)
    d_board = f.create_dataset(
        "board_t",
        shape=(0, BOARD_HEIGHT, BOARD_WIDTH, BOARD_CHANNELS),
        maxshape=(None, BOARD_HEIGHT, BOARD_WIDTH, BOARD_CHANNELS),
        dtype=np.float32,
        chunks=ch,
    )
    d_pos = f.create_dataset(
        "board_t_plus_1_pos",
        shape=(0, BOARD_HEIGHT, BOARD_WIDTH, BOARD_CHANNELS),
        maxshape=(None, BOARD_HEIGHT, BOARD_WIDTH, BOARD_CHANNELS),
        dtype=np.float32,
        chunks=ch,
    )
    d_neg = f.create_dataset(
        "board_t_plus_1_negs",
        shape=(0, k_neg, BOARD_HEIGHT, BOARD_WIDTH, BOARD_CHANNELS),
        maxshape=(None, k_neg, BOARD_HEIGHT, BOARD_WIDTH, BOARD_CHANNELS),
        dtype=np.float32,
        chunks=(1, k_neg, BOARD_HEIGHT, BOARD_WIDTH, BOARD_CHANNELS),
    )
    d_elo = f.create_dataset("elo", shape=(0,), maxshape=(None,), dtype=np.float32, chunks=(1024,))
    dsets = {
        "board_t": d_board,
        "board_t_plus_1_pos": d_pos,
        "board_t_plus_1_negs": d_neg,
        "elo": d_elo,
    }
    return f, dsets


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert PGN to HDF5 train/val/test for Chess-JEPA transitions."
    )
    parser.add_argument("pgn", type=Path, help="Input PGN file")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for train.h5, val.h5, test.h5 (default: same dir as PGN)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for split and sampling")
    parser.add_argument(
        "--num-negatives",
        type=int,
        default=NUM_NEGATIVES_DEFAULT,
        metavar="K",
        help=f"Number of negative next-states per row (default {NUM_NEGATIVES_DEFAULT})",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        metavar="N",
        help="Stop after this many rows total (train+val+test). No limit if unset.",
    )
    args = parser.parse_args()
    k_neg = args.num_negatives
    if k_neg < 1:
        print("Error: --num-negatives must be >= 1", file=sys.stderr)
        return 1

    if not args.pgn.exists():
        print(f"Error: PGN file not found: {args.pgn}", file=sys.stderr)
        return 1
    if args.max_rows is not None and args.max_rows < 1:
        print("Error: --max-rows must be >= 1", file=sys.stderr)
        return 1

    out_dir = args.output_dir or args.pgn.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    def empty_bufs() -> dict[str, list]:
        return {"board_t": [], "board_t_plus_1_pos": [], "board_t_plus_1_negs": [], "elo": []}

    train_bufs = empty_bufs()
    val_bufs = empty_bufs()
    test_bufs = empty_bufs()

    train_file = out_dir / "train.h5"
    val_file = out_dir / "val.h5"
    test_file = out_dir / "test.h5"

    train_h5: h5py.File | None = None
    val_h5: h5py.File | None = None
    test_h5: h5py.File | None = None
    train_ds: dict[str, h5py.Dataset] = {}
    val_ds: dict[str, h5py.Dataset] = {}
    test_ds: dict[str, h5py.Dataset] = {}
    train_next = 0
    val_next = 0
    test_next = 0

    total_train = total_val = total_test = 0
    game_count = 0

    with open(args.pgn, encoding="utf-8", errors="replace") as fp:
        pbar = tqdm(desc="Games", unit=" games", dynamic_ncols=True, file=sys.stderr)
        while True:
            game = chess.pgn.read_game(fp)
            if game is None:
                break
            game_count += 1
            split = _assign_split(rng)
            t, v, te = _process_game(
                game,
                rng,
                k_neg,
                train_bufs,
                val_bufs,
                test_bufs,
                split,
            )
            total_train += t
            total_val += v
            total_test += te
            pbar.update(1)
            pbar.set_postfix(train=total_train, val=total_val, test=total_test, refresh=False)

            if args.max_rows is not None and (total_train + total_val + total_test) >= args.max_rows:
                break

            if len(train_bufs["board_t"]) >= HDF5_FLUSH_BATCH_SIZE:
                if train_h5 is None:
                    train_h5, train_ds = _create_file(train_file, k_neg)
                    train_next = 0
                train_next = _flush_jepa_batch(train_h5, train_bufs, train_ds, train_next, k_neg)
            if len(val_bufs["board_t"]) >= HDF5_FLUSH_BATCH_SIZE:
                if val_h5 is None:
                    val_h5, val_ds = _create_file(val_file, k_neg)
                    val_next = 0
                val_next = _flush_jepa_batch(val_h5, val_bufs, val_ds, val_next, k_neg)
            if len(test_bufs["board_t"]) >= HDF5_FLUSH_BATCH_SIZE:
                if test_h5 is None:
                    test_h5, test_ds = _create_file(test_file, k_neg)
                    test_next = 0
                test_next = _flush_jepa_batch(test_h5, test_bufs, test_ds, test_next, k_neg)

    pbar.close()

    if train_bufs["board_t"]:
        if train_h5 is None:
            train_h5, train_ds = _create_file(train_file, k_neg)
            train_next = 0
        train_next = _flush_jepa_batch(train_h5, train_bufs, train_ds, train_next, k_neg)
    if val_bufs["board_t"]:
        if val_h5 is None:
            val_h5, val_ds = _create_file(val_file, k_neg)
            val_next = 0
        val_next = _flush_jepa_batch(val_h5, val_bufs, val_ds, val_next, k_neg)
    if test_bufs["board_t"]:
        if test_h5 is None:
            test_h5, test_ds = _create_file(test_file, k_neg)
            test_next = 0
        test_next = _flush_jepa_batch(test_h5, test_bufs, test_ds, test_next, k_neg)

    if train_h5:
        train_h5.close()
    if val_h5:
        val_h5.close()
    if test_h5:
        test_h5.close()

    print(file=sys.stderr)
    print(f"Games processed: {game_count}", file=sys.stderr)
    print(f"Train: {total_train} rows -> {train_file}", file=sys.stderr)
    print(f"Val:   {total_val} rows -> {val_file}", file=sys.stderr)
    print(f"Test:  {total_test} rows -> {test_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
