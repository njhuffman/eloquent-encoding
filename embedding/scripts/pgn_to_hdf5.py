#!/usr/bin/env python3
"""
Convert a PGN file to HDF5 datasets (train/val/test) with board tensors and metadata.
Each game is assigned to one split (80/10/10). Positions are sampled within games
with opening skip and per-move skip probabilities. Writes in batches to support
millions of boards.
"""

import argparse
import random
import sys
from pathlib import Path

import chess
import chess.pgn
import h5py
import numpy as np
from tqdm import tqdm

# Add repo root for embedding imports
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from embedding.board_encoding import board_to_tensor
from embedding.config import (
    BOARD_CHANNELS,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    FIRST_10_MOVES_SKIP_PROB,
    HDF5_FLUSH_BATCH_SIZE,
    OPENING_SKIP_HALF_MOVES,
    REMAINING_MOVES_SKIP_PROB,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)


def _parse_result(result_str: str) -> int:
    """Map PGN Result to outcome: 1 white win, 0 draw, -1 black win."""
    if result_str == "1-0":
        return 1
    if result_str == "0-1":
        return -1
    if result_str == "1/2-1/2" or result_str == "*":
        return 0
    return 0  # unknown treat as draw


def _parse_elo(headers: chess.pgn.Headers, color: str) -> int:
    """Get Elo from headers; return 0 if missing."""
    key = "WhiteElo" if color == "white" else "BlackElo"
    try:
        return int(headers.get(key, 0) or 0)
    except (ValueError, TypeError):
        return 0


def _piece_count(board: chess.Board, color: chess.Color) -> int:
    return sum(1 for p in board.piece_map().values() if p.color == color)


def _process_game(
    game: chess.pgn.Game,
    rng: random.Random,
    train_boards: list,
    train_meta: list,
    val_boards: list,
    val_meta: list,
    test_boards: list,
    test_meta: list,
    split: str,
) -> tuple[int, int, int]:
    """Replay game, sample positions, append to the appropriate split buffers. Returns (train_added, val_added, test_added)."""
    board = game.board()
    headers = game.headers
    result_str = headers.get("Result", "*")
    outcome = _parse_result(result_str)
    elo_white = _parse_elo(headers, "white")
    elo_black = _parse_elo(headers, "black")

    train_added = val_added = test_added = 0
    half_move = 0

    for move in game.mainline_moves():
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

        tensor = board_to_tensor(board)
        pc_white = _piece_count(board, chess.WHITE)
        pc_black = _piece_count(board, chess.BLACK)
        in_check = 1 if board.is_check() else 0
        meta_row = [float(elo_white), float(elo_black), float(pc_white), float(pc_black), float(outcome), float(in_check)]

        if split == "train":
            train_boards.append(tensor)
            train_meta.append(meta_row)
            train_added += 1
        elif split == "val":
            val_boards.append(tensor)
            val_meta.append(meta_row)
            val_added += 1
        else:
            test_boards.append(tensor)
            test_meta.append(meta_row)
            test_added += 1

    return train_added, val_added, test_added


def _assign_split(rng: random.Random) -> str:
    r = rng.random()
    if r < TRAIN_RATIO:
        return "train"
    if r < TRAIN_RATIO + VAL_RATIO:
        return "val"
    return "test"


def _flush_batch(
    f: h5py.File,
    boards: list,
    meta: list,
    board_dataset: h5py.Dataset,
    meta_dataset: h5py.Dataset,
    next_idx: int,
) -> int:
    if not boards:
        return next_idx
    n = len(boards)
    arr = np.stack(boards, axis=0)
    meta_arr = np.array(meta, dtype=np.float32)
    new_size = next_idx + n
    board_dataset.resize(new_size, axis=0)
    meta_dataset.resize(new_size, axis=0)
    board_dataset[next_idx:new_size] = arr
    meta_dataset[next_idx:new_size] = meta_arr
    return new_size


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert PGN to HDF5 train/val/test with board tensors and metadata."
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
        "--max-boards",
        type=int,
        default=None,
        metavar="N",
        help="Stop after this many boards total (train+val+test). No limit if unset.",
    )
    args = parser.parse_args()

    if not args.pgn.exists():
        print(f"Error: PGN file not found: {args.pgn}", file=sys.stderr)
        return 1
    if args.max_boards is not None and args.max_boards < 1:
        print("Error: --max-boards must be >= 1", file=sys.stderr)
        return 1

    out_dir = args.output_dir or args.pgn.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    # Buffers per split
    train_boards: list = []
    train_meta: list = []
    val_boards: list = []
    val_meta: list = []
    test_boards: list = []
    test_meta: list = []

    # We'll create HDF5 files on first flush and extend datasets
    train_file = out_dir / "train.h5"
    val_file = out_dir / "val.h5"
    test_file = out_dir / "test.h5"

    train_next = val_next = test_next = 0
    train_h5 = val_h5 = test_h5 = None
    train_ds = val_ds = test_ds = None
    train_mds = val_mds = test_mds = None

    def ensure_train():
        nonlocal train_h5, train_ds, train_mds, train_next
        if train_h5 is None:
            train_h5 = h5py.File(train_file, "w")
            train_ds = train_h5.create_dataset(
                "board",
                shape=(0, BOARD_HEIGHT, BOARD_WIDTH, BOARD_CHANNELS),
                maxshape=(None, BOARD_HEIGHT, BOARD_WIDTH, BOARD_CHANNELS),
                dtype=np.float32,
                chunks=(1, BOARD_HEIGHT, BOARD_WIDTH, BOARD_CHANNELS),
            )
            train_mds = train_h5.create_dataset(
                "meta",
                shape=(0, 6),
                maxshape=(None, 6),
                dtype=np.float32,
            )

    def ensure_val():
        nonlocal val_h5, val_ds, val_mds, val_next
        if val_h5 is None:
            val_h5 = h5py.File(val_file, "w")
            val_ds = val_h5.create_dataset(
                "board",
                shape=(0, BOARD_HEIGHT, BOARD_WIDTH, BOARD_CHANNELS),
                maxshape=(None, BOARD_HEIGHT, BOARD_WIDTH, BOARD_CHANNELS),
                dtype=np.float32,
                chunks=(1, BOARD_HEIGHT, BOARD_WIDTH, BOARD_CHANNELS),
            )
            val_mds = val_h5.create_dataset("meta", shape=(0, 6), maxshape=(None, 6), dtype=np.float32)

    def ensure_test():
        nonlocal test_h5, test_ds, test_mds, test_next
        if test_h5 is None:
            test_h5 = h5py.File(test_file, "w")
            test_ds = test_h5.create_dataset(
                "board",
                shape=(0, BOARD_HEIGHT, BOARD_WIDTH, BOARD_CHANNELS),
                maxshape=(None, BOARD_HEIGHT, BOARD_WIDTH, BOARD_CHANNELS),
                dtype=np.float32,
                chunks=(1, BOARD_HEIGHT, BOARD_WIDTH, BOARD_CHANNELS),
            )
            test_mds = test_h5.create_dataset("meta", shape=(0, 6), maxshape=(None, 6), dtype=np.float32)

    total_train = total_val = total_test = 0
    game_count = 0

    with open(args.pgn, encoding="utf-8", errors="replace") as f:
        pbar = tqdm(desc="Games", unit=" games", dynamic_ncols=True, file=sys.stderr)
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            game_count += 1
            split = _assign_split(rng)
            t, v, te = _process_game(
                game,
                rng,
                train_boards,
                train_meta,
                val_boards,
                val_meta,
                test_boards,
                test_meta,
                split,
            )
            total_train += t
            total_val += v
            total_test += te
            pbar.update(1)
            pbar.set_postfix(train=total_train, val=total_val, test=total_test, refresh=False)

            if args.max_boards is not None and (total_train + total_val + total_test) >= args.max_boards:
                break

            # Flush when any buffer reaches batch size
            if len(train_boards) >= HDF5_FLUSH_BATCH_SIZE:
                ensure_train()
                train_next = _flush_batch(train_h5, train_boards, train_meta, train_ds, train_mds, train_next)
                train_boards.clear()
                train_meta.clear()
            if len(val_boards) >= HDF5_FLUSH_BATCH_SIZE:
                ensure_val()
                val_next = _flush_batch(val_h5, val_boards, val_meta, val_ds, val_mds, val_next)
                val_boards.clear()
                val_meta.clear()
            if len(test_boards) >= HDF5_FLUSH_BATCH_SIZE:
                ensure_test()
                test_next = _flush_batch(test_h5, test_boards, test_meta, test_ds, test_mds, test_next)
                test_boards.clear()
                test_meta.clear()

    pbar.close()

    # Final flush
    if train_boards:
        ensure_train()
        _flush_batch(train_h5, train_boards, train_meta, train_ds, train_mds, train_next)
    if val_boards:
        ensure_val()
        _flush_batch(val_h5, val_boards, val_meta, val_ds, val_mds, val_next)
    if test_boards:
        ensure_test()
        _flush_batch(test_h5, test_boards, test_meta, test_ds, test_mds, test_next)

    if train_h5:
        train_h5.close()
    if val_h5:
        val_h5.close()
    if test_h5:
        test_h5.close()

    print(file=sys.stderr)
    print(f"Games processed: {game_count}", file=sys.stderr)
    print(f"Train: {total_train} boards -> {train_file}", file=sys.stderr)
    print(f"Val:   {total_val} boards -> {val_file}", file=sys.stderr)
    print(f"Test:  {total_test} boards -> {test_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
