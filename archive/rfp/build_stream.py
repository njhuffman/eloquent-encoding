"""
PGN → rfp HDF5: precomputed encoder embeddings + gfp-compatible labels.
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
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset_generation.candidate_collect import board_at_ply, collect_candidate_positions
from dataset_generation.pgn_prefilter import any_unfilled_stratum_may_match, game_matches_stratum
from dataset_generation.recipe import Recipe, SourcePlan, StratumSpec
from dataset_generation.resolve import resolve_source_file
from dataset_generation.stream import iter_filtered_pgn_game_texts_from_zstd

from embedding.board_encoding import board_to_tensor
from gfp.encoder import load_jepa3_encoder_from_checkpoint
from jepa3.board_masks import legal_from_square_mask
from jepa3.packed_board_codec import legal_mask_float_to_u64
from move_predictor.encoding import move_to_from_to
from rfp.h5_io import DATASET_DELTA_Z, RfpH5Writer

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


def _boards_to_z_batch(
    encoder: torch.nn.Module,
    boards_np: list[np.ndarray],
    device: torch.device,
) -> np.ndarray:
    """boards_np: list of (8,8,18) float32 → (C, d_model) float32 numpy."""
    if not boards_np:
        return np.zeros((0, 0), dtype=np.float32)
    batch = np.stack(boards_np, axis=0)
    t = torch.from_numpy(batch).to(device)
    with torch.inference_mode():
        z = encoder(t)
    return z.detach().float().cpu().numpy()


def _pack_deltas(z_stack: np.ndarray, history_len: int) -> tuple[np.ndarray, np.ndarray]:
    """
    z_stack: (C, D) consecutive global embeddings from oldest to newest pre-move board.
    Returns delta_z (history_len, D) float16, history_mask (history_len,) uint8.
    """
    n = int(history_len)
    d = int(z_stack.shape[1])
    if z_stack.shape[0] < 2:
        dz = np.zeros((n, d), dtype=np.float16)
        mask = np.zeros((n,), dtype=np.uint8)
        return dz, mask
    rd = z_stack[1:].astype(np.float32) - z_stack[:-1].astype(np.float32)
    L = rd.shape[0]
    dz = np.zeros((n, d), dtype=np.float32)
    mask = np.zeros((n,), dtype=np.uint8)
    if L >= n:
        dz[:] = rd[-n:]
        mask[:] = 1
    else:
        dz[n - L :] = rd
        mask[n - L :] = 1
    return dz.astype(np.float16), mask


def _write_samples_rfp(
    writer: RfpH5Writer,
    encoder: torch.nn.Module,
    device: torch.device,
    history_len: int,
    mainline: list[chess.Move],
    candidates: list[tuple[int, int, int, chess.Move]],
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
            "internal error: _write_samples_rfp requires "
            f"len(candidates) >= samples_per_game ({k} < {stratum.samples_per_game})"
        )
    rng = _rng_for_game(master_seed, source_plan_index, stratum, stratum_index, g)
    for j in _sample_indices(rng, k, stratum.samples_per_game):
        ply, _stm, elo_tm, move = candidates[j]
        board = board_at_ply(mainline, ply)
        fr, _to = move_to_from_to(move)
        if __debug__:
            if move not in board.legal_moves:
                raise RuntimeError(f"internal error: illegal move in sample at ply {ply}: {move}")
        from_mask = legal_from_square_mask(board)
        fu = legal_mask_float_to_u64(from_mask)

        start_ply = max(0, ply - history_len)
        boards_np = [board_to_tensor(board_at_ply(mainline, pp)) for pp in range(start_ply, ply + 1)]
        z_stack = _boards_to_z_batch(encoder, boards_np, device)
        delta_z, hist_mask = _pack_deltas(z_stack, history_len)
        z_curr = z_stack[-1].astype(np.float16)

        elo_bucket = int(np.clip(elo_tm // 100, -32768, 32767))

        writer.append_row(
            delta_z=delta_z,
            z_curr=z_curr,
            history_mask=hist_mask,
            from_legal_u64=fu,
            from_sq=int(fr),
            elo_bucket=elo_bucket,
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
    writer: RfpH5Writer,
    encoder: torch.nn.Module,
    device: torch.device,
    history_len: int,
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
                _write_samples_rfp(
                    writer,
                    encoder,
                    device,
                    history_len,
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


def build_rfp_from_recipe(
    recipe: Recipe,
    *,
    data_dir: Path,
    output_dir: Path,
    encoder_checkpoint: Path | str,
    history_len: int,
    device: torch.device | None = None,
    encoder_strict: bool = True,
) -> Path:
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_jepa3_encoder_from_checkpoint(
        encoder_checkpoint,
        device=dev,
        strict=bool(encoder_strict),
    )
    d_model = int(encoder.d_model)

    out_dir = output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = recipe.output_h5_path(output_dir)
    if out_path.exists():
        out_path.unlink()

    try:
        with RfpH5Writer(out_path, history_len=int(history_len), d_model=d_model) as writer:
            for plan_index, plan in enumerate(recipe.source_plans):
                _process_one_source_plan(
                    writer,
                    encoder,
                    dev,
                    int(history_len),
                    recipe,
                    plan_index=plan_index,
                    plan=plan,
                    data_dir=data_dir,
                )
    except Exception as e:
        if out_path.exists():
            out_path.unlink(missing_ok=True)
            logger.error("RFP build failed; removed partial output %s: %s", out_path, e)
        else:
            logger.error("RFP build failed (no output file written): %s", e)
        raise

    expected = recipe.target_sample_rows()
    with h5py.File(out_path, "r") as f:
        n = int(f[DATASET_DELTA_Z].shape[0])
    if n != expected:
        out_path.unlink(missing_ok=True)
        msg = f"HDF5 row count {n} != recipe target {expected}; removed output"
        logger.error(msg)
        raise RuntimeError(msg)
    return out_path
