from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

import chess
import chess.pgn
import h5py
import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset_generation.move_encode import move_to_from_to, promotion_code
from dataset_generation.hdf5_io import PackedBatchWriter
from style_policy.board_encode import board_to_packed, legal_from_u64, legal_to_u64

from dataset_generation.candidate_collect import board_at_ply, collect_candidate_positions
from dataset_generation.pgn_prefilter import any_unfilled_stratum_may_match, game_matches_stratum
from dataset_generation.recipe import Recipe, SourcePlan, StratumSpec
from dataset_generation.resolve import resolve_source_file
from dataset_generation.stream import iter_filtered_pgn_game_texts_from_zstd


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
    """Requires ``k >= samples_per_game`` (caller ensures after filtering short games)."""
    idx = rng.choice(k, size=samples_per_game, replace=False)
    return np.sort(idx)


def _write_samples_for_stratum(
    writer: PackedBatchWriter,
    mainline: list[chess.Move],
    candidates: list[tuple[int, int, int, int, int, chess.Move]],
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
            "internal error: _write_samples_for_stratum requires "
            f"len(candidates) >= samples_per_game ({k} < {stratum.samples_per_game})"
        )
    rng = _rng_for_game(master_seed, source_plan_index, stratum, stratum_index, g)
    for j in _sample_indices(rng, k, stratum.samples_per_game):
        ply, stm, elo, opp_elo, result, move, hist = candidates[j]
        board = board_at_ply(mainline, ply)
        fr, to = move_to_from_to(move)
        writer.append_row(
            packed_pre=board_to_packed(board),
            from_legal_u64=legal_from_u64(board),
            to_legal_u64=legal_to_u64(board, fr),
            from_sq=fr, to_sq=to, promotion=promotion_code(move),
            elo_to_move=int(elo), opp_elo=int(opp_elo), result=int(result),
            hist_from=[h[0] for h in hist],
            hist_to=[h[1] for h in hist],
            hist_cap=[h[2] for h in hist],
        )


def _ensure_strata_quotas_met(plan: SourcePlan, accepted: list[int]) -> None:
    """Raise if the stream ended before any stratum reached its take_games quota."""
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
    writer: PackedBatchWriter,
    recipe: Recipe,
    *,
    plan_index: int,
    plan: SourcePlan,
    data_dir: Path,
) -> list[int]:
    """Stream `plan.source` once; fill that plan's strata in parallel. Only games with
    at least ``samples_per_game`` candidate positions count toward each stratum's
    ``take_games``. Raises if EOF arrives before every quota is met."""
    strata = plan.strata
    accepted = [0] * len(strata)
    path = resolve_source_file(data_dir, plan.source)
    try:
        raw = open(path, "rb")
    except OSError as e:
        raise RuntimeError(f"failed to open source {plan.source!r}: {e}") from e
    try:
        game_iter = iter_filtered_pgn_game_texts_from_zstd(
            raw, recipe=recipe, plan=plan, accepted=accepted
        )
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

            if not any_unfilled_stratum_may_match(
                white, black, recipe=recipe, plan=plan, accepted=accepted
            ):
                continue

            mainline, candidates = collect_candidate_positions(
                game,
                skip_opening_plies=recipe.skip_opening_plies,
                exclude_single_legal_move=recipe.exclude_single_legal_move,
            )

            for s, st in enumerate(strata):
                if accepted[s] >= st.take_games:
                    continue
                if not game_matches_stratum(
                    white, black, recipe.bucket_by, st.elo_min, st.elo_max
                ):
                    continue
                if len(candidates) < st.samples_per_game:
                    continue
                g = accepted[s]
                accepted[s] += 1
                _write_samples_for_stratum(
                    writer,
                    mainline,
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


def build_from_recipe(
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
        with PackedBatchWriter(out_path) as writer:
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
            logger.error("Build failed; removed partial output %s: %s", out_path, e)
        else:
            logger.error("Build failed (no output file written): %s", e)
        raise

    expected = recipe.target_sample_rows()
    with h5py.File(out_path, "r") as f:
        n = int(f["packed_pre"].shape[0])
    if n != expected:
        out_path.unlink(missing_ok=True)
        msg = f"HDF5 row count {n} != recipe target {expected}; removed output"
        logger.error(msg)
        raise RuntimeError(msg)
    return out_path
