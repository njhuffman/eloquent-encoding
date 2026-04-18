"""
Build JEPA-format training tensors in RAM from move-sample HDF5 + optional hard-negative mining.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

import chess
import h5py
import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from jepa.architectures.chess_jepa_v1 import ChessJEPA
from jepa.config import BOARD_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH
from jepa.move_row_codec import row_to_board_and_move, tensor_after_move, tensors_for_row

FLUSH_EVERY = 2048

JepaSplitArrays = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def estimate_materialized_split_bytes(n_rows: int, k_neg: int) -> int:
    """Upper-bound bytes for float32 JEPA tensors for ``n_rows`` positions (train or val split)."""
    if n_rows < 0:
        raise ValueError("n_rows must be non-negative")
    elem = BOARD_HEIGHT * BOARD_WIDTH * BOARD_CHANNELS
    per_row = 4 * int((2 + k_neg) * elem + 1)
    return int(n_rows) * per_row


def _append_buf_blocks(blocks: dict[str, list[np.ndarray]], bufs: dict[str, list]) -> None:
    if not bufs["board_t"]:
        return
    blocks["board_t"].append(np.stack(bufs["board_t"], axis=0).astype(np.float32, copy=False))
    blocks["board_t_plus_1_pos"].append(np.stack(bufs["board_t_plus_1_pos"], axis=0).astype(np.float32, copy=False))
    blocks["board_t_plus_1_negs"].append(np.stack(bufs["board_t_plus_1_negs"], axis=0))
    blocks["elo"].append(np.stack(bufs["elo"], axis=0).astype(np.float32, copy=False))
    for k in bufs:
        bufs[k].clear()


def _concat_blocks(blocks: dict[str, list[np.ndarray]], k_neg: int) -> JepaSplitArrays:
    h, w, c = BOARD_HEIGHT, BOARD_WIDTH, BOARD_CHANNELS
    if not blocks["board_t"]:
        bt = np.empty((0, h, w, c), dtype=np.float32)
        pos = np.empty((0, h, w, c), dtype=np.float32)
        negs = np.empty((0, k_neg, h, w, c), dtype=np.float32)
        elo = np.empty((0,), dtype=np.float32)
        return bt, pos, negs, elo
    bt = np.concatenate(blocks["board_t"], axis=0)
    pos = np.concatenate(blocks["board_t_plus_1_pos"], axis=0)
    negs = np.concatenate(blocks["board_t_plus_1_negs"], axis=0)
    elo = np.concatenate(blocks["elo"], axis=0)
    return bt, pos, negs, elo


def _random_neg_tensors(
    board: chess.Board,
    true_move: chess.Move,
    k: int,
    rng: random.Random,
) -> list[np.ndarray] | None:
    wrong = [m for m in board.legal_moves if m != true_move]
    if not wrong:
        return None
    out: list[np.ndarray] = []
    for _ in range(k):
        out.append(tensor_after_move(board, rng.choice(wrong)))
    return out


def _eval_legal_subset(
    legals: list[chess.Move],
    true_move: chess.Move,
    max_n: int | None,
    rng: random.Random,
) -> tuple[list[chess.Move], int]:
    """
    Moves to score with the target encoder. If max_n is None or len(legals) <= max_n,
    use all legals. Otherwise sample max_n moves including true_move, then shuffle.
    Returns (eval_legals, true_j index into eval_legals).
    """
    if max_n is None or len(legals) <= max_n:
        ev = list(legals)
        return ev, ev.index(true_move)
    others = [m for m in legals if m != true_move]
    picked = [true_move] + rng.sample(others, max_n - 1)
    rng.shuffle(picked)
    return picked, picked.index(true_move)


def _pick_hard_and_random(
    sims: np.ndarray,
    true_j: int,
    legal_moves: list[chess.Move],
    board: chess.Board,
    true_move: chess.Move,
    n_hard: int,
    m_random: int,
    k_neg: int,
    rng: random.Random,
) -> list[np.ndarray] | None:
    """sims[j] higher = z_hat closer to successor j (e.g. -||z_t(j)-z_hat||^2); true_j index of true move."""
    wrong_idx = [j for j in range(len(legal_moves)) if legal_moves[j] != true_move]
    if len(wrong_idx) < k_neg:
        return None

    scored = sorted(wrong_idx, key=lambda j: float(sims[j]), reverse=True)
    hard_js = scored[:n_hard]
    pool = [j for j in wrong_idx if j not in hard_js]
    need_random = k_neg - len(hard_js)
    random_js: list[int] = []
    if need_random > 0:
        if len(pool) >= need_random:
            random_js = rng.sample(pool, need_random)
        elif pool:
            random_js = [rng.choice(pool) for _ in range(need_random)]
        else:
            return None

    chosen_js = hard_js + random_js
    if len(chosen_js) != k_neg:
        return None
    return [tensor_after_move(board, legal_moves[j]) for j in chosen_js]


def materialize_jepa_split(
    move_h5_path: Path,
    row_indices: np.ndarray,
    *,
    progress_desc: str = "materialize",
    model: ChessJEPA | None,
    device: torch.device,
    k_neg: int,
    n_hard: int,
    m_random: int,
    use_hard_mining: bool,
    neg_seed: int,
    mining_position_batch: int = 64,
    evaluate_legals_n: int | None = None,
) -> tuple[dict[str, Any], JepaSplitArrays]:
    """
    Build JEPA arrays in RAM. model may be None only if not use_hard_mining.
    evaluate_legals_n: when hard mining, score at most this many legal moves (None = all).
    """
    if n_hard + m_random != k_neg:
        raise ValueError(f"n_hard + m_random must equal k_neg ({k_neg})")
    if use_hard_mining and model is None:
        raise ValueError("use_hard_mining requires a model")

    rng_py = random.Random(neg_seed)
    n_skip = 0
    n_written = 0

    bufs: dict[str, list] = {
        "board_t": [],
        "board_t_plus_1_pos": [],
        "board_t_plus_1_negs": [],
        "elo": [],
    }
    blocks: dict[str, list[np.ndarray]] = {
        "board_t": [],
        "board_t_plus_1_pos": [],
        "board_t_plus_1_negs": [],
        "elo": [],
    }

    with h5py.File(move_h5_path, "r") as f_m:
        n_total_rows = int(f_m["fen"].shape[0])

        def read_row(i: int) -> tuple:
            fen = f_m["fen"][i]
            if isinstance(fen, bytes):
                fen = fen.decode("utf-8")
            elo = float(f_m["elo_to_move"][i])
            fs = int(f_m["from_sq"][i])
            ts = int(f_m["to_sq"][i])
            pr = int(f_m["promotion"][i])
            return str(fen), elo, fs, ts, pr

        pos = 0
        pbar = tqdm(total=len(row_indices), desc=progress_desc, unit="rows")

        try:
            while pos < len(row_indices):
                end = min(pos + mining_position_batch, len(row_indices))
                chunk_idx = row_indices[pos:end]

                batch_items: list[
                    tuple[np.ndarray, np.ndarray, chess.Board, chess.Move, float, list[chess.Move], int]
                ] = []

                for ii in chunk_idx:
                    ii = int(ii)
                    if ii < 0 or ii >= n_total_rows:
                        n_skip += 1
                        continue
                    fen, elo, fs, ts, pr = read_row(ii)
                    t = tensors_for_row(fen, fs, ts, pr)
                    if t is None:
                        n_skip += 1
                        continue
                    bt, pos_t = t
                    parsed = row_to_board_and_move(fen, fs, ts, pr)
                    assert parsed is not None
                    board, move_true = parsed
                    legals = list(board.legal_moves)
                    if len(legals) < 2:
                        n_skip += 1
                        continue
                    try:
                        true_j = legals.index(move_true)
                    except ValueError:
                        n_skip += 1
                        continue
                    batch_items.append((bt, pos_t, board, move_true, elo, legals, true_j))

                if use_hard_mining and model is not None and batch_items:
                    model.eval()
                    with torch.no_grad():
                        bt_stack = torch.from_numpy(
                            np.stack([x[0] for x in batch_items], axis=0)
                        ).to(device)
                        elo_t = torch.tensor(
                            [x[4] for x in batch_items], dtype=torch.float32, device=device
                        )
                        _, z_hat = model.forward_online(bt_stack, elo_t)
                        z_hat_np = z_hat.float().cpu().numpy()

                    for bi, item in enumerate(batch_items):
                        bt, pos_t, board, move_true, elo, legals, _true_j_full = item
                        eval_legals, true_j = _eval_legal_subset(
                            legals, move_true, evaluate_legals_n, rng_py
                        )
                        succ = np.stack(
                            [tensor_after_move(board, m) for m in eval_legals],
                            axis=0,
                        ).astype(np.float32)
                        st = torch.from_numpy(succ).to(device)
                        with torch.no_grad():
                            z_t = model.forward_target(st)
                            z_t_np = z_t.float().cpu().numpy()
                        zh = z_hat_np[bi]
                        sims = -np.sum((z_t_np - zh.reshape(1, -1)) ** 2, axis=-1)
                        neg_tensors = _pick_hard_and_random(
                            sims,
                            true_j,
                            eval_legals,
                            board,
                            move_true,
                            n_hard,
                            m_random,
                            k_neg,
                            rng_py,
                        )
                        if neg_tensors is None:
                            n_skip += 1
                            continue
                        bufs["board_t"].append(bt)
                        bufs["board_t_plus_1_pos"].append(pos_t)
                        bufs["board_t_plus_1_negs"].append(np.stack(neg_tensors, axis=0))
                        bufs["elo"].append(np.float32(elo))
                        n_written += 1
                        if len(bufs["board_t"]) >= FLUSH_EVERY:
                            _append_buf_blocks(blocks, bufs)

                elif batch_items:
                    for item in batch_items:
                        bt, pos_t, board, move_true, elo, legals, _true_j = item
                        neg_tensors = _random_neg_tensors(board, move_true, k_neg, rng_py)
                        if neg_tensors is None:
                            n_skip += 1
                            continue
                        bufs["board_t"].append(bt)
                        bufs["board_t_plus_1_pos"].append(pos_t)
                        bufs["board_t_plus_1_negs"].append(np.stack(neg_tensors, axis=0))
                        bufs["elo"].append(np.float32(elo))
                        n_written += 1
                        if len(bufs["board_t"]) >= FLUSH_EVERY:
                            _append_buf_blocks(blocks, bufs)

                pbar.update(len(chunk_idx))
                pos = end

            _append_buf_blocks(blocks, bufs)
        finally:
            pbar.close()

    arrays = _concat_blocks(blocks, k_neg)
    report: dict[str, Any] = {
        "n_written": n_written,
        "n_skip": n_skip,
        "storage": "ram",
        "use_hard_mining": use_hard_mining,
    }
    return report, arrays


def train_pool_indices(
    n_total: int,
    val_set: set[int],
    n_train: int,
    seed: int,
) -> np.ndarray:
    pool = [i for i in range(n_total) if i not in val_set]
    rng = np.random.default_rng(seed)
    take = min(n_train, len(pool))
    return rng.choice(np.array(pool, dtype=np.int64), size=take, replace=False)


def val_indices(n_total: int, n_val: int, seed: int) -> np.ndarray:
    take = min(n_val, n_total)
    rng = np.random.default_rng(seed)
    return rng.choice(n_total, size=take, replace=False)
