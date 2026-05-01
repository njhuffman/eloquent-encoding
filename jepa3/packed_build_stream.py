"""
PGN → packed HDF5 row streaming.

Duplicated from dataset_generation/builder.py (same sampling / quotas); writes jepa3
PackedMoveH5Writer rows instead of fen-based SampleBatchWriter. Keep in sync with
dataset_generation/builder.py when changing sampling rules.
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import chess
import chess.pgn
import h5py
import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from embedding.board_encoding import board_to_tensor
from jepa.move_row_codec import tensor_after_move
from jepa3.board_masks import legal_from_square_mask, legal_to_square_mask
from jepa3.packed_board_codec import board_tensor_to_packed, legal_masks_to_u64
from jepa3.packed_h5 import PackedMoveH5Writer
from move_predictor.encoding import move_to_from_to, promotion_code

from dataset_generation.recipe import Recipe, SourcePlan, StratumSpec
from dataset_generation.resolve import resolve_source_file
from dataset_generation.stream import iter_pgn_games_from_zstd_binary

logger = logging.getLogger(__name__)


def _parse_elo(headers: chess.pgn.Headers, color: str) -> int | None:
    key = f"{color.capitalize()}Elo"
    raw = headers.get(key)
    if raw is None or raw == "?":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _game_matches_time_control(game: chess.pgn.Game, required: str | None) -> bool:
    if required is None:
        return True
    tc = game.headers.get("TimeControl")
    return tc == required


def _game_matches_stratum(
    white: int | None,
    black: int | None,
    bucket_by: str,
    lo: int,
    hi: int,
) -> bool:
    if white is None or black is None:
        return False
    if bucket_by == "white":
        return lo <= white <= hi
    if bucket_by == "black":
        return lo <= black <= hi
    if bucket_by == "both":
        return lo <= white <= hi and lo <= black <= hi
    raise ValueError(bucket_by)


def collect_candidate_positions(
    game: chess.pgn.Game,
    *,
    skip_opening_plies: int,
    exclude_single_legal_move: bool,
) -> list[tuple[str, int, int, chess.Move]]:
    white_elo_h = _parse_elo(game.headers, "white")
    black_elo_h = _parse_elo(game.headers, "black")
    if white_elo_h is None or black_elo_h is None:
        return []

    board = game.board()
    ply = 0
    out: list[tuple[str, int, int, chess.Move]] = []
    for move in game.mainline_moves():
        if ply >= skip_opening_plies:
            n_legal = board.legal_moves.count()
            if (not exclude_single_legal_move) or n_legal >= 2:
                fe = board.fen()
                stm = 0 if board.turn == chess.WHITE else 1
                elo_tm = white_elo_h if board.turn == chess.WHITE else black_elo_h
                out.append((fe, stm, elo_tm, move))
        board.push(move)
        ply += 1
    return out


def _rng_for_game(
    master_seed: int,
    source_plan_index: int,
    stratum: StratumSpec,
    stratum_index: int,
    g: int,
) -> np.random.Generator:
    ss = np.random.SeedSequence(
        [master_seed, source_plan_index, stratum.stratum_seed, stratum_index, g]
    )
    return np.random.Generator(np.random.PCG64(ss))


def _sample_indices(
    rng: np.random.Generator,
    k: int,
    samples_per_game: int,
) -> np.ndarray:
    idx = rng.choice(k, size=samples_per_game, replace=False)
    return np.sort(idx)


def _write_samples_packed(
    writer: PackedMoveH5Writer,
    candidates: list[tuple[str, int, int, chess.Move]],
    *,
    master_seed: int,
    source_plan_index: int,
    stratum: StratumSpec,
    stratum_index: int,
    g: int,
) -> None:
    k = len(candidates)
    if k < stratum.samples_per_game:
        raise RuntimeError(
            "internal error: _write_samples_packed requires "
            f"len(candidates) >= samples_per_game ({k} < {stratum.samples_per_game})"
        )
    rng = _rng_for_game(master_seed, source_plan_index, stratum, stratum_index, g)
    for j in _sample_indices(rng, k, stratum.samples_per_game):
        fen, _stm, elo, move = candidates[j]
        board = chess.Board(fen)
        fr, to = move_to_from_to(move)
        pr = promotion_code(move)
        if move not in board.legal_moves:
            raise RuntimeError(f"internal error: illegal move in sample: {fen!r} {move}")
        t_pre = board_to_tensor(board)
        t_post = tensor_after_move(board, move)
        fm = legal_from_square_mask(board)
        tm = legal_to_square_mask(board, fr)
        fu, tu = legal_masks_to_u64(fm, tm)
        writer.append_row(
            packed_pre=board_tensor_to_packed(t_pre),
            packed_post=board_tensor_to_packed(t_post),
            from_legal_u64=fu,
            to_legal_u64=tu,
            from_sq=fr,
            to_sq=to,
            promotion=pr,
            elo_to_move=int(elo),
        )


def _ensure_strata_quotas_met(plan: SourcePlan, accepted: list[int]) -> None:
    short: list[str] = []
    for s, st in enumerate(plan.strata):
        got = accepted[s]
        if got < st.take_games:
            short.append(
                f"stratum[{s}] elo [{st.elo_min},{st.elo_max}]: "
                f"need take_games={st.take_games}, got {got}"
            )
    if short:
        msg = (
            f"source {plan.source!r} ended before quotas were met:\n  - "
            + "\n  - ".join(short)
        )
        logger.error(msg)
        raise RuntimeError(msg)


def _process_one_source_plan(
    writer: PackedMoveH5Writer,
    recipe: Recipe,
    *,
    plan_index: int,
    plan: SourcePlan,
    data_dir: Path,
) -> list[int]:
    strata = plan.strata
    accepted = [0] * len(strata)
    path = resolve_source_file(data_dir, plan.source)
    try:
        raw = open(path, "rb")
    except OSError as e:
        raise RuntimeError(f"failed to open source {plan.source!r}: {e}") from e
    try:
        game_iter = iter_pgn_games_from_zstd_binary(raw)
        pbar = tqdm(game_iter, desc=path.name, unit=" games")
        for text in pbar:
            if all(accepted[s] >= strata[s].take_games for s in range(len(strata))):
                break

            game = chess.pgn.read_game(io.StringIO(text))
            if game is None:
                continue
            if not _game_matches_time_control(game, recipe.time_control):
                continue

            white = _parse_elo(game.headers, "white")
            black = _parse_elo(game.headers, "black")

            candidates = collect_candidate_positions(
                game,
                skip_opening_plies=recipe.skip_opening_plies,
                exclude_single_legal_move=recipe.exclude_single_legal_move,
            )

            for s, st in enumerate(strata):
                if accepted[s] >= st.take_games:
                    continue
                if not _game_matches_stratum(white, black, recipe.bucket_by, st.elo_min, st.elo_max):
                    continue
                if len(candidates) < st.samples_per_game:
                    continue
                g = accepted[s]
                accepted[s] += 1
                _write_samples_packed(
                    writer,
                    candidates,
                    master_seed=recipe.master_seed,
                    source_plan_index=plan_index,
                    stratum=st,
                    stratum_index=s,
                    g=g,
                )

            pbar.set_postfix({f"s{i}": accepted[i] for i in range(len(accepted))}, refresh=False)
    finally:
        if hasattr(raw, "close"):
            raw.close()
    _ensure_strata_quotas_met(plan, accepted)
    return accepted


def build_packed_from_recipe(
    recipe: Recipe,
    *,
    data_dir: Path,
    output_dir: Path,
) -> Path:
    out_dir = output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = recipe.output_h5_path(output_dir)
    if out_path.exists():
        out_path.unlink()

    try:
        with PackedMoveH5Writer(out_path) as writer:
            for plan_index, plan in enumerate(recipe.source_plans):
                _process_one_source_plan(
                    writer,
                    recipe,
                    plan_index=plan_index,
                    plan=plan,
                    data_dir=data_dir,
                )
    except Exception as e:
        if out_path.exists():
            out_path.unlink(missing_ok=True)
            logger.error("Packed build failed; removed partial output %s: %s", out_path, e)
        else:
            logger.error("Packed build failed (no output file written): %s", e)
        raise

    expected = recipe.target_sample_rows()
    from jepa3.packed_h5 import DATASET_PACKED_PRE

    with h5py.File(out_path, "r") as f:
        n = int(f[DATASET_PACKED_PRE].shape[0])
    if n != expected:
        out_path.unlink(missing_ok=True)
        msg = f"HDF5 row count {n} != recipe target {expected}; removed output"
        logger.error(msg)
        raise RuntimeError(msg)
    return out_path
