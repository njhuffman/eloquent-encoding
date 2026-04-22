#!/usr/bin/env python3
"""
Benchmark Chess-JEPA checkpoints on move prediction: rank all legal moves by latent
distance to the predictor output (same score as hard-negative mining in materialize).

Move-sample HDF5 schema: ``fen``, ``from_sq``, ``to_sq``, ``promotion``, ``elo_to_move``.

Example::

    python -m jepa.scripts.prediction_benchmark \\
        --checkpoints jepa_checkpoints/foo/best.pt jepa_checkpoints/bar/best.pt \\
        --move-h5 databases/moves/validation_moves_1M.h5 \\
        --sample-n 5000 \\
        --seed 42

Per-checkpoint output includes the same top-k / rank stats broken down by ``elo_to_move``
(default 100-point buckets; override with ``--elo-bucket-width``).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from jepa.h5_bootstrap import apply_hdf5_read_safety_env

apply_hdf5_read_safety_env()

from embedding.board_encoding import board_to_tensor
from jepa.load import load_jepa_from_checkpoint
from jepa.move_row_codec import row_to_board_and_move, tensor_after_move


ELO_BUCKET_WIDTH = 100


@dataclass
class RankMetrics:
    """Move-ranking aggregates for a subset of positions (no skip counts)."""

    n_positions: int
    top1_pct: float
    top2_pct: float
    top3_pct: float
    top5_pct: float
    top10_pct: float
    mean_rank: float
    median_rank: float
    mean_reciprocal_rank: float
    mean_normalized_rank: float
    mean_n_legals: float


@dataclass
class BenchmarkStats:
    """Aggregates over valid positions (>=2 legal moves, parseable row, true move legal)."""

    n_positions: int
    n_skipped_parse: int
    n_skipped_few_legals: int
    n_skipped_true_not_legal: int
    top1_pct: float
    top2_pct: float
    top3_pct: float
    top5_pct: float
    top10_pct: float
    mean_rank: float
    median_rank: float
    mean_reciprocal_rank: float
    mean_normalized_rank: float
    mean_n_legals: float


@dataclass
class PositivePredStats:
    """Median Euclidean ‖z_hat − z_pos‖ with z_pos from EMA target vs online encoder (same rows as move benchmark)."""

    n_positions: int
    n_skipped_parse: int
    n_skipped_few_legals: int
    n_skipped_true_not_legal: int
    median_l2_pred_to_pos_ema: float
    median_l2_pred_to_pos_online: float


def _read_move_row(f: h5py.File, i: int) -> tuple[str, float, int, int, int]:
    fen = f["fen"][i]
    if isinstance(fen, bytes):
        fen = fen.decode("utf-8")
    elo = float(f["elo_to_move"][i])
    fs = int(f["from_sq"][i])
    ts = int(f["to_sq"][i])
    pr = int(f["promotion"][i])
    return str(fen), elo, fs, ts, pr


def _rank_from_scores(sims: np.ndarray, true_j: int) -> int:
    """1-based rank; higher sim is better. Ties: count strictly better scores only."""
    st = float(sims[true_j])
    return int(np.sum(sims > st) + 1)


def _top_k_hit(rank: int, k: int, n_legals: int) -> bool:
    if n_legals < 2:
        return False
    k_eff = min(k, n_legals)
    return rank <= k_eff


def _elo_bucket_start(elo: float, *, width: int = ELO_BUCKET_WIDTH) -> int:
    """Lower bound of half-open [start, start+width) for player-to-move Elo."""
    return int(np.floor(float(elo) / float(width))) * int(width)


def _compute_rank_metrics(ranks: list[int], n_legals_list: list[int]) -> RankMetrics:
    n_pos = len(ranks)
    assert n_pos == len(n_legals_list) and n_pos > 0
    r = np.asarray(ranks, dtype=np.float64)
    nl = np.asarray(n_legals_list, dtype=np.float64)

    def pct_for_k(k: int) -> float:
        return float(
            np.mean([_top_k_hit(int(ranks[i]), k, int(n_legals_list[i])) for i in range(n_pos)]) * 100.0
        )

    norm = (r - 1.0) / np.maximum(nl - 1.0, 1.0)
    return RankMetrics(
        n_positions=n_pos,
        top1_pct=pct_for_k(1),
        top2_pct=pct_for_k(2),
        top3_pct=pct_for_k(3),
        top5_pct=pct_for_k(5),
        top10_pct=pct_for_k(10),
        mean_rank=float(np.mean(r)),
        median_rank=float(np.median(r)),
        mean_reciprocal_rank=float(np.mean(1.0 / r)),
        mean_normalized_rank=float(np.mean(norm)),
        mean_n_legals=float(np.mean(nl)),
    )


def _sample_indices(n_total: int, sample_n: int, seed: int) -> np.ndarray:
    n = min(int(sample_n), n_total)
    if n <= 0:
        return np.array([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=n, replace=False))


def _score_legals(
    model: torch.nn.Module,
    board_t: np.ndarray,
    elo: float,
    succ_stack: np.ndarray,
    from_sq: int,
    device: torch.device,
    *,
    use_amp: bool,
    succ_chunk: int,
) -> np.ndarray:
    """Return sims (L,) = negative squared L2 between z_hat and each z_target(successor)."""
    sims, _, _ = _sims_zhat_positive_l2(
        model,
        board_t,
        elo,
        succ_stack,
        None,
        device,
        from_sq,
        use_amp=use_amp,
        succ_chunk=succ_chunk,
    )
    return sims


def _sims_zhat_positive_l2(
    model: torch.nn.Module,
    board_t: np.ndarray,
    elo: float,
    succ_stack: np.ndarray,
    board_pos: np.ndarray | None,
    device: torch.device,
    from_sq: int,
    *,
    use_amp: bool,
    succ_chunk: int,
) -> tuple[np.ndarray, float | None, float | None]:
    """
    One ``forward_online`` per position, chunked ``forward_target`` on legals, optional
    true-successor L2 ‖z_hat - z_pos‖ (EMA vs online positive encoder).

    ``from_sq`` is the dataset row's starting square index (0..63), passed into
    ``forward_online`` so architectures that condition the predictor on from-square
    see the known square at benchmark time (matches eval, not the unknown token).
    """
    model.eval()
    bt = torch.from_numpy(board_t).unsqueeze(0).to(device)
    elo_t = torch.tensor([elo], dtype=torch.float32, device=device)
    from_sq_t = torch.tensor([int(from_sq)], dtype=torch.long, device=device)
    l_total = succ_stack.shape[0]
    sims_list: list[np.ndarray] = []
    l2_ema: float | None = None
    l2_on: float | None = None

    with torch.no_grad():
        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                _, z_hat = model.forward_online(bt, elo_t, from_sq_t)
        else:
            _, z_hat = model.forward_online(bt, elo_t, from_sq_t)
        z_hat_f = z_hat.float()

        for start in range(0, l_total, succ_chunk):
            end = min(start + succ_chunk, l_total)
            chunk = torch.from_numpy(succ_stack[start:end]).to(device)
            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    z_t = model.forward_target(chunk)
            else:
                z_t = model.forward_target(chunk)
            z_t = z_t.float()
            zh = z_hat_f.expand(z_t.shape[0], -1)
            d = (z_t - zh).pow(2).sum(dim=-1)
            sims_list.append((-d).cpu().numpy())

        if board_pos is not None and hasattr(model, "encoder_online"):
            bp = torch.from_numpy(board_pos).unsqueeze(0).to(device)
            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    z_pos_ema = model.forward_target(bp)
                    z_pos_on = model.encoder_online(bp)
            else:
                z_pos_ema = model.forward_target(bp)
                z_pos_on = model.encoder_online(bp)
            z_pos_ema = z_pos_ema.float().squeeze(0)
            z_pos_on = z_pos_on.float().squeeze(0)
            zh = z_hat_f.squeeze(0)
            l2_ema = float(torch.sqrt((zh - z_pos_ema).pow(2).sum()).item())
            l2_on = float(torch.sqrt((zh - z_pos_on).pow(2).sum()).item())

    return np.concatenate(sims_list, axis=0), l2_ema, l2_on


def run_move_and_positive_metrics_for_checkpoint(
    ckpt_path: Path,
    f: h5py.File,
    indices: np.ndarray,
    device: torch.device,
    *,
    use_amp: bool,
    succ_chunk: int,
    elo_bucket_width: int = ELO_BUCKET_WIDTH,
    quiet: bool = False,
    load_model: Callable[[Path, torch.device], Any] | None = None,
) -> tuple[BenchmarkStats, dict[int, RankMetrics], PositivePredStats]:
    """Move-ranking benchmark plus median pred–positive L2 (EMA vs online positive); single model load and one pass per row."""
    if load_model is not None:
        model = load_model(ckpt_path, device)
    else:
        model = load_jepa_from_checkpoint(ckpt_path, device=device)
    track_positive = hasattr(model, "encoder_online") and hasattr(model, "forward_target")

    ranks: list[int] = []
    n_legals_list: list[int] = []
    elos: list[float] = []
    l2_ema_list: list[float] = []
    l2_on_list: list[float] = []
    n_skipped_parse = 0
    n_skipped_few_legals = 0
    n_skipped_true_not_legal = 0

    iterator = indices if quiet else tqdm(indices, desc=f"eval {ckpt_path.name}", unit="pos")
    for ii in iterator:
        i = int(ii)
        fen, elo, fs, ts, pr = _read_move_row(f, i)
        parsed = row_to_board_and_move(fen, fs, ts, pr)
        if parsed is None:
            n_skipped_parse += 1
            continue
        board, move_true = parsed
        legals = list(board.legal_moves)
        if len(legals) < 2:
            n_skipped_few_legals += 1
            continue
        try:
            true_j = legals.index(move_true)
        except ValueError:
            n_skipped_true_not_legal += 1
            continue

        board_t = board_to_tensor(board).astype(np.float32, copy=False)
        succ_stack = np.stack([tensor_after_move(board, m) for m in legals], axis=0)
        board_pos = (
            tensor_after_move(board, move_true).astype(np.float32, copy=False) if track_positive else None
        )

        sims, l2_e, l2_o = _sims_zhat_positive_l2(
            model,
            board_t,
            elo,
            succ_stack,
            board_pos,
            device,
            fs,
            use_amp=use_amp,
            succ_chunk=succ_chunk,
        )
        rank = _rank_from_scores(sims, true_j)
        ranks.append(rank)
        n_legals_list.append(len(legals))
        elos.append(float(elo))
        if track_positive and l2_e is not None and l2_o is not None:
            l2_ema_list.append(l2_e)
            l2_on_list.append(l2_o)

    n_pos = len(ranks)
    if n_pos == 0:
        empty_b = BenchmarkStats(
            n_positions=0,
            n_skipped_parse=n_skipped_parse,
            n_skipped_few_legals=n_skipped_few_legals,
            n_skipped_true_not_legal=n_skipped_true_not_legal,
            top1_pct=0.0,
            top2_pct=0.0,
            top3_pct=0.0,
            top5_pct=0.0,
            top10_pct=0.0,
            mean_rank=float("nan"),
            median_rank=float("nan"),
            mean_reciprocal_rank=float("nan"),
            mean_normalized_rank=float("nan"),
            mean_n_legals=float("nan"),
        )
        empty_p = PositivePredStats(
            n_positions=0,
            n_skipped_parse=n_skipped_parse,
            n_skipped_few_legals=n_skipped_few_legals,
            n_skipped_true_not_legal=n_skipped_true_not_legal,
            median_l2_pred_to_pos_ema=float("nan"),
            median_l2_pred_to_pos_online=float("nan"),
        )
        return empty_b, {}, empty_p

    rm = _compute_rank_metrics(ranks, n_legals_list)
    by_bucket: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for rank, nl, elo in zip(ranks, n_legals_list, elos):
        b0 = _elo_bucket_start(elo, width=elo_bucket_width)
        by_bucket[b0].append((rank, nl))

    bucket_metrics: dict[int, RankMetrics] = {}
    for b0 in sorted(by_bucket.keys()):
        pairs = by_bucket[b0]
        br = [p[0] for p in pairs]
        bn = [p[1] for p in pairs]
        bucket_metrics[b0] = _compute_rank_metrics(br, bn)

    stats = BenchmarkStats(
        n_positions=rm.n_positions,
        n_skipped_parse=n_skipped_parse,
        n_skipped_few_legals=n_skipped_few_legals,
        n_skipped_true_not_legal=n_skipped_true_not_legal,
        top1_pct=rm.top1_pct,
        top2_pct=rm.top2_pct,
        top3_pct=rm.top3_pct,
        top5_pct=rm.top5_pct,
        top10_pct=rm.top10_pct,
        mean_rank=rm.mean_rank,
        median_rank=rm.median_rank,
        mean_reciprocal_rank=rm.mean_reciprocal_rank,
        mean_normalized_rank=rm.mean_normalized_rank,
        mean_n_legals=rm.mean_n_legals,
    )

    if l2_ema_list:
        med_e = float(np.median(np.asarray(l2_ema_list, dtype=np.float64)))
        med_o = float(np.median(np.asarray(l2_on_list, dtype=np.float64)))
        pos_stats = PositivePredStats(
            n_positions=len(l2_ema_list),
            n_skipped_parse=n_skipped_parse,
            n_skipped_few_legals=n_skipped_few_legals,
            n_skipped_true_not_legal=n_skipped_true_not_legal,
            median_l2_pred_to_pos_ema=med_e,
            median_l2_pred_to_pos_online=med_o,
        )
    else:
        pos_stats = PositivePredStats(
            n_positions=0,
            n_skipped_parse=n_skipped_parse,
            n_skipped_few_legals=n_skipped_few_legals,
            n_skipped_true_not_legal=n_skipped_true_not_legal,
            median_l2_pred_to_pos_ema=float("nan"),
            median_l2_pred_to_pos_online=float("nan"),
        )

    return stats, bucket_metrics, pos_stats


def run_benchmark_for_checkpoint(
    ckpt_path: Path,
    f: h5py.File,
    indices: np.ndarray,
    device: torch.device,
    *,
    use_amp: bool,
    succ_chunk: int,
    elo_bucket_width: int = ELO_BUCKET_WIDTH,
    quiet: bool = False,
) -> tuple[BenchmarkStats, dict[int, RankMetrics]]:
    stats, by_elo, _ = run_move_and_positive_metrics_for_checkpoint(
        ckpt_path,
        f,
        indices,
        device,
        use_amp=use_amp,
        succ_chunk=succ_chunk,
        elo_bucket_width=elo_bucket_width,
        quiet=quiet,
    )
    return stats, by_elo


def main() -> int:
    parser = argparse.ArgumentParser(description="JEPA move-prediction benchmark over legal moves")
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        required=True,
        help="One or more JEPA checkpoint paths (.pt)",
    )
    parser.add_argument(
        "--move-h5",
        type=Path,
        required=True,
        help="Move-sample HDF5 (fen, from_sq, to_sq, promotion, elo_to_move)",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        required=True,
        help="Number of dataset rows to evaluate (random subset; capped by file size)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for row subsample")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="torch device (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Use CUDA autocast for forwards (matches training)",
    )
    parser.add_argument(
        "--succ-chunk",
        type=int,
        default=256,
        help="Max legal successors per forward_target chunk (memory vs speed)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write all results as JSON",
    )
    parser.add_argument(
        "--elo-bucket-width",
        type=int,
        default=ELO_BUCKET_WIDTH,
        help="Elo bucket width for per-bucket stats (player to move; default 100)",
    )
    args = parser.parse_args()
    elo_bw = max(1, int(args.elo_bucket_width))

    move_h5 = args.move_h5.resolve()
    if not move_h5.is_file():
        print(f"Error: move HDF5 not found: {move_h5}", file=sys.stderr)
        return 1

    for p in args.checkpoints:
        if not p.resolve().is_file():
            print(f"Error: checkpoint not found: {p}", file=sys.stderr)
            return 1

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    with h5py.File(move_h5, "r") as f:
        for key in ("fen", "from_sq", "to_sq", "promotion", "elo_to_move"):
            if key not in f:
                print(f"Error: {move_h5} missing dataset {key!r}", file=sys.stderr)
                return 1
        n_total = int(f["fen"].shape[0])
        indices = _sample_indices(n_total, args.sample_n, args.seed)
        print(
            f"Dataset rows={n_total}, evaluating n={len(indices)} (seed={args.seed}), device={device}",
            file=sys.stderr,
        )

        results: dict[str, dict] = {}
        for ckpt in args.checkpoints:
            ckpt = ckpt.resolve()
            stats, by_elo = run_benchmark_for_checkpoint(
                ckpt,
                f,
                indices,
                device,
                use_amp=args.use_amp,
                succ_chunk=max(1, int(args.succ_chunk)),
                elo_bucket_width=elo_bw,
            )
            results[str(ckpt)] = {
                **asdict(stats),
                "by_elo_bucket": {str(b0): asdict(m) for b0, m in sorted(by_elo.items())},
            }

    print("\n=== JEPA move prediction benchmark ===\n")
    print(
        f"{'checkpoint':<48} "
        f"{'n':>5} "
        f"top1% top2% top3% top5% top10% "
        f"mean_rank  mrr mean_norm mean_legals"
    )
    for ckpt_str, d in results.items():
        name = Path(ckpt_str).name
        disp = name if len(name) <= 46 else name[:43] + "..."
        print(
            f"{disp:<48} "
            f"{d['n_positions']:5d} "
            f"{d['top1_pct']:5.1f} {d['top2_pct']:5.1f} {d['top3_pct']:5.1f} "
            f"{d['top5_pct']:5.1f} {d['top10_pct']:5.1f} "
            f"{d['mean_rank']:8.2f} {d['mean_reciprocal_rank']:5.3f} "
            f"{d['mean_normalized_rank']:8.3f} {d['mean_n_legals']:6.1f}"
        )

    print(f"\n=== By Elo to move ({elo_bw}-point buckets) ===\n")
    hdr = (
        f"{'elo_range':<12} "
        f"{'n':>5} "
        f"top1% top2% top3% top5% top10% "
        f"mean_rank  mrr mean_norm mean_legals"
    )
    for ckpt_str, d in results.items():
        name = Path(ckpt_str).name
        print(f"{name}")
        print(hdr)
        buckets = d.get("by_elo_bucket") or {}
        if not buckets:
            print("  (no positions)\n")
            continue
        for bkey in sorted(buckets.keys(), key=lambda x: int(x)):
            m = buckets[bkey]
            b0 = int(bkey)
            hi = b0 + elo_bw - 1
            label = f"{b0}-{hi}"
            print(
                f"{label:<12} "
                f"{m['n_positions']:5d} "
                f"{m['top1_pct']:5.1f} {m['top2_pct']:5.1f} {m['top3_pct']:5.1f} "
                f"{m['top5_pct']:5.1f} {m['top10_pct']:5.1f} "
                f"{m['mean_rank']:8.2f} {m['mean_reciprocal_rank']:5.3f} "
                f"{m['mean_normalized_rank']:8.3f} {m['mean_n_legals']:6.1f}"
            )
        print()

    print("\nSkipped: parse / few_legals / true_not_in_legals")
    for ckpt_str, d in results.items():
        name = Path(ckpt_str).name
        print(
            f"  {name}: {d['n_skipped_parse']} / {d['n_skipped_few_legals']} / {d['n_skipped_true_not_legal']}"
        )

    if args.json_out is not None:
        out = {
            "move_h5": str(move_h5),
            "sample_n_requested": args.sample_n,
            "seed": args.seed,
            "indices_len": int(len(indices)),
            "elo_bucket_width": elo_bw,
            "checkpoints": results,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
