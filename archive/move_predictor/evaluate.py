#!/usr/bin/env python3
"""
Evaluate a move-predictor checkpoint by ranking every legal move vs the played move.

Unlike training (3-way CE among a few candidates), this scores the full legal set
and reports top-k hit rates, mean rank, and MRR. Optionally compares to a random
baseline (independent random score per legal move, same ranking protocol).

Inputs (pick one):
  --test-h5   Move-predictor HDF5 (dual color histories, side_to_move, fen, label, from/to/prom).
  --pgn       Replay games; rebuild history like pgn_to_move_hdf5 (needs MAE for embeddings).
  --fen-file  One position per line: FEN and UCI of the played move; both color histories empty.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
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

from embedding.board_encoding import board_to_tensor
from embedding.load import load_mae_by_name, load_mae_from_checkpoint
from move_predictor.model import MovePredictor


def _load_move_predictor(ckpt: Path, device: torch.device) -> MovePredictor:
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


def _chosen_move(
    from_sq: np.ndarray,
    to_sq: np.ndarray,
    prom: np.ndarray,
    label: int,
) -> chess.Move:
    p = int(prom[label])
    return chess.Move(
        int(from_sq[label]),
        int(to_sq[label]),
        promotion=chess.PieceType(p) if p != 0 else None,
    )


def _encoder_input_from_board(board: np.ndarray) -> np.ndarray:
    from embedding.config import BOARD_HEIGHT, BOARD_WIDTH

    mask = np.zeros((BOARD_HEIGHT, BOARD_WIDTH, 1), dtype=np.float32)
    return np.concatenate([board.astype(np.float32), mask], axis=-1)


def _encode_boards(
    mae: torch.nn.Module,
    boards: list[np.ndarray],
    device: torch.device,
) -> np.ndarray:
    if not boards:
        return np.zeros((0, int(mae.embedding_dim)), dtype=np.float32)
    from embedding.config import BOARD_HEIGHT, BOARD_WIDTH

    enc_in = np.stack([_encoder_input_from_board(b) for b in boards], axis=0)
    with torch.no_grad():
        x = torch.from_numpy(enc_in).to(device)
        if x.shape[-1] == 19:
            x = x.permute(0, 3, 1, 2)
        emb = mae.encoder(x)
        return emb.cpu().numpy().astype(np.float32)


def _rank_legal_scores(
    scores: torch.Tensor,
    legal: list[chess.Move],
    played: chess.Move,
) -> tuple[int, int] | None:
    """Return (1-based rank, num legal) or None if played not legal."""
    try:
        played_idx = legal.index(played)
    except ValueError:
        return None
    order = torch.argsort(scores, descending=True, stable=True)
    pos = int(torch.where(order == played_idx)[0].item())
    return pos + 1, len(legal)


def rank_position_random(
    rng: np.random.Generator,
    board: chess.Board,
    played: chess.Move,
) -> tuple[int, int] | None:
    """Random predictor: i.i.d. standard normal score per legal move (unique ordering a.s.)."""
    legal = list(board.legal_moves)
    if not legal:
        return None
    scores = torch.tensor(rng.standard_normal(len(legal)), dtype=torch.float32)
    return _rank_legal_scores(scores, legal, played)


def rank_position(
    model: MovePredictor,
    device: torch.device,
    cur_emb: torch.Tensor,
    hist_w: torch.Tensor,
    hist_b: torch.Tensor,
    hlen_w: torch.Tensor,
    hlen_b: torch.Tensor,
    side_to_move: torch.Tensor,
    board: chess.Board,
    played: chess.Move,
    *,
    chunk_k: int,
) -> tuple[int, int] | None:
    """
    cur_emb: (1, D); hist_*: (1, N, D); hlen_* / side_to_move: (1,) long.
    """
    legal = list(board.legal_moves)
    if not legal:
        return None
    k = len(legal)
    parts: list[torch.Tensor] = []
    for start in range(0, k, chunk_k):
        end = min(start + chunk_k, k)
        sl = legal[start:end]
        fs = torch.tensor([[m.from_square for m in sl]], dtype=torch.long, device=device)
        ts = torch.tensor([[m.to_square for m in sl]], dtype=torch.long, device=device)
        with torch.no_grad():
            parts.append(
                model.score_moves(
                    cur_emb, hist_w, hist_b, hlen_w, hlen_b, side_to_move, fs, ts
                )[0]
            )
    scores = torch.cat(parts, dim=0)
    return _rank_legal_scores(scores, legal, played)


def _parse_fen_file_line(line: str) -> tuple[str, str] | None:
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    for sep in ("\t", "|"):
        if sep in s:
            a, b = s.split(sep, 1)
            a, b = a.strip(), b.strip()
            return (a, b) if b else None
    parts = s.split()
    if len(parts) >= 5:
        return " ".join(parts[:-1]), parts[-1]
    return None


class Agg:
    def __init__(self) -> None:
        self.n = 0
        self.skipped_not_legal = 0
        self.sum_k = 0
        self.sum_rank = 0.0
        self.sum_recip = 0.0
        self.top = {1: 0, 3: 0, 5: 0, 10: 0}

    def add(self, rank: int, k: int) -> None:
        self.n += 1
        self.sum_k += k
        self.sum_rank += rank
        self.sum_recip += 1.0 / rank
        for t in self.top:
            if rank <= t:
                self.top[t] += 1

    def report(self, *, heading: str | None = None, file=sys.stdout) -> None:
        if heading:
            print(heading, file=file)
        if self.n == 0:
            print("No positions evaluated.", file=file)
            return
        print(f"positions_evaluated={self.n}  skipped_played_not_legal={self.skipped_not_legal}", file=file)
        print(f"mean_legal_moves={self.sum_k / self.n:.2f}", file=file)
        print(f"mean_rank={self.sum_rank / self.n:.3f}  mrr={self.sum_recip / self.n:.4f}", file=file)
        for t in sorted(self.top):
            pct = 100.0 * self.top[t] / self.n
            print(f"top_{t}_acc={pct:.2f}% ({self.top[t]}/{self.n})", file=file)


def eval_h5(
    path: Path,
    model: MovePredictor,
    device: torch.device,
    *,
    chunk_k: int,
    limit: int | None,
    agg_random: Agg | None,
    rng: np.random.Generator | None,
) -> Agg:
    agg = Agg()
    with h5py.File(path, "r") as f:
        n = int(f["cur_emb"].shape[0])
        if limit is not None:
            n = min(n, limit)
        has_prom = "promotion" in f
        for i in tqdm(range(n), desc="H5 rows", file=sys.stderr):
            fen_raw = f["fen"][i]
            if isinstance(fen_raw, bytes):
                fen_raw = fen_raw.decode("utf-8")
            board = chess.Board(fen_raw)
            label = int(f["label"][i])
            ff = np.asarray(f["from_sq"][i], dtype=np.int64)
            tt = np.asarray(f["to_sq"][i], dtype=np.int64)
            if has_prom:
                pp = np.asarray(f["promotion"][i], dtype=np.int64)
            else:
                pp = np.zeros(3, dtype=np.int64)
            played = _chosen_move(ff, tt, pp, label)

            cur = torch.from_numpy(np.asarray(f["cur_emb"][i], dtype=np.float32)).unsqueeze(0).to(device)
            hw = torch.from_numpy(np.asarray(f["hist_white_emb"][i], dtype=np.float32)).unsqueeze(0).to(device)
            hb = torch.from_numpy(np.asarray(f["hist_black_emb"][i], dtype=np.float32)).unsqueeze(0).to(device)
            lw = torch.tensor([int(f["hist_white_len"][i])], dtype=torch.long, device=device)
            lb = torch.tensor([int(f["hist_black_len"][i])], dtype=torch.long, device=device)
            turn = torch.tensor([int(f["side_to_move"][i])], dtype=torch.long, device=device)

            out = rank_position(
                model, device, cur, hw, hb, lw, lb, turn, board, played, chunk_k=chunk_k
            )
            if out is None:
                agg.skipped_not_legal += 1
                continue
            r, k = out
            agg.add(r, k)
            if agg_random is not None and rng is not None:
                out_r = rank_position_random(rng, board, played)
                if out_r is not None:
                    agg_random.add(*out_r)
    return agg


def eval_pgn(
    path: Path,
    model: MovePredictor,
    mae: torch.nn.Module,
    device: torch.device,
    *,
    history_n: int,
    embedding_dim: int,
    chunk_k: int,
    limit_games: int | None,
    agg_random: Agg | None,
    rng: np.random.Generator | None,
) -> Agg:
    agg = Agg()
    games_done = 0
    pbar = tqdm(desc="PGN positions", unit="pos", file=sys.stderr)
    with open(path, encoding="utf-8", errors="replace") as fp:
        while True:
            if limit_games is not None and games_done >= limit_games:
                break
            game = chess.pgn.read_game(fp)
            if game is None:
                break
            games_done += 1
            board = game.board()
            prev_white: deque[chess.Board] = deque(maxlen=history_n)
            prev_black: deque[chess.Board] = deque(maxlen=history_n)
            for move in game.mainline_moves():
                legal = list(board.legal_moves)
                if not legal:
                    sk = board.copy()
                    board.push(move)
                    if sk.turn == chess.WHITE:
                        prev_white.append(sk)
                    else:
                        prev_black.append(sk)
                    pbar.update(1)
                    continue

                hw_boards = list(prev_white)
                hb_boards = list(prev_black)
                n_w, n_b = len(hw_boards), len(hb_boards)
                cur_t = board_to_tensor(board)
                cur_emb_np = _encode_boards(mae, [cur_t], device)[0]
                hw_np = np.zeros((history_n, embedding_dim), dtype=np.float32)
                if n_w:
                    hw_np[:n_w] = _encode_boards(
                        mae, [board_to_tensor(b) for b in hw_boards], device
                    )
                hb_np = np.zeros((history_n, embedding_dim), dtype=np.float32)
                if n_b:
                    hb_np[:n_b] = _encode_boards(
                        mae, [board_to_tensor(b) for b in hb_boards], device
                    )

                cur = torch.from_numpy(cur_emb_np).unsqueeze(0).to(device)
                hw_t = torch.from_numpy(hw_np).unsqueeze(0).to(device)
                hb_t = torch.from_numpy(hb_np).unsqueeze(0).to(device)
                lw_t = torch.tensor([n_w], dtype=torch.long, device=device)
                lb_t = torch.tensor([n_b], dtype=torch.long, device=device)
                turn_t = torch.tensor(
                    [0 if board.turn == chess.WHITE else 1], dtype=torch.long, device=device
                )

                out = rank_position(
                    model, device, cur, hw_t, hb_t, lw_t, lb_t, turn_t, board, move, chunk_k=chunk_k
                )
                if out is None:
                    agg.skipped_not_legal += 1
                else:
                    agg.add(*out)
                    if agg_random is not None and rng is not None:
                        out_r = rank_position_random(rng, board, move)
                        if out_r is not None:
                            agg_random.add(*out_r)
                pbar.update(1)

                sk = board.copy()
                board.push(move)
                if sk.turn == chess.WHITE:
                    prev_white.append(sk)
                else:
                    prev_black.append(sk)
    pbar.close()
    return agg


def eval_fen_file(
    path: Path,
    model: MovePredictor,
    mae: torch.nn.Module,
    device: torch.device,
    *,
    history_n: int,
    embedding_dim: int,
    chunk_k: int,
    limit: int | None,
    agg_random: Agg | None,
    rng: np.random.Generator | None,
) -> Agg:
    agg = Agg()
    hw_np = np.zeros((history_n, embedding_dim), dtype=np.float32)
    hb_np = np.zeros((history_n, embedding_dim), dtype=np.float32)
    hw = torch.from_numpy(hw_np).unsqueeze(0).to(device)
    hb = torch.from_numpy(hb_np).unsqueeze(0).to(device)
    hlen_zero = torch.tensor([0], dtype=torch.long, device=device)

    n_eval = 0
    pbar = tqdm(desc="FEN lines", unit="pos", file=sys.stderr)
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if limit is not None and n_eval >= limit:
                break
            parsed = _parse_fen_file_line(line)
            if parsed is None:
                continue
            fen_s, uci_s = parsed
            n_eval += 1
            board = chess.Board(fen_s)
            try:
                played = chess.Move.from_uci(uci_s.strip())
            except ValueError:
                agg.skipped_not_legal += 1
                pbar.update(1)
                continue

            cur_t = board_to_tensor(board)
            cur_emb_np = _encode_boards(mae, [cur_t], device)[0]
            cur = torch.from_numpy(cur_emb_np).unsqueeze(0).to(device)
            turn_t = torch.tensor(
                [0 if board.turn == chess.WHITE else 1], dtype=torch.long, device=device
            )

            out = rank_position(
                model,
                device,
                cur,
                hw,
                hb,
                hlen_zero,
                hlen_zero,
                turn_t,
                board,
                played,
                chunk_k=chunk_k,
            )
            if out is None:
                agg.skipped_not_legal += 1
                pbar.update(1)
                continue
            agg.add(*out)
            if agg_random is not None and rng is not None:
                out_r = rank_position_random(rng, board, played)
                if out_r is not None:
                    agg_random.add(*out_r)
            pbar.update(1)
    pbar.close()
    return agg


def main() -> int:
    parser = argparse.ArgumentParser(description="Full legal-set move prediction evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True, help="move_predictor .pt checkpoint")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--test-h5", type=Path, help="Move HDF5 (embeddings + fen + label/from/to/prom)")
    src.add_argument("--pgn", type=Path, help="PGN games (replays positions; needs MAE)")
    src.add_argument(
        "--fen-file",
        type=Path,
        help="Lines: FEN<TAB>UCI or FEN|UCI (empty white/black history; needs MAE)",
    )
    parser.add_argument(
        "--mae-checkpoint",
        type=Path,
        default=None,
        help="Board encoder checkpoint (required for --pgn and --fen-file)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Registered MAE name (alternative to --mae-checkpoint)",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--chunk-k",
        type=int,
        default=256,
        help="Max legal moves scored per forward (memory vs speed)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max rows (--test-h5 / --fen-file) or games (--pgn)")
    parser.add_argument(
        "--no-baseline-random",
        action="store_true",
        help="Skip random-score baseline (i.i.d. per legal move)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="RNG seed for random baseline (default: 42)",
    )
    args = parser.parse_args()

    if args.pgn is not None or args.fen_file is not None:
        if args.mae_checkpoint is None and args.embedding_model is None:
            print("Error: --pgn / --fen-file require --mae-checkpoint or --embedding-model", file=sys.stderr)
            return 1

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _load_move_predictor(args.checkpoint, device)
    do_random = not args.no_baseline_random
    rng = np.random.default_rng(args.random_seed) if do_random else None
    agg_random = Agg() if do_random else None

    if args.test_h5 is not None:
        if not args.test_h5.is_file():
            print(f"Error: not found {args.test_h5}", file=sys.stderr)
            return 1
        agg = eval_h5(
            args.test_h5,
            model,
            device,
            chunk_k=args.chunk_k,
            limit=args.limit,
            agg_random=agg_random,
            rng=rng,
        )
    else:
        mae: torch.nn.Module
        if args.mae_checkpoint is not None:
            mae = load_mae_from_checkpoint(args.mae_checkpoint, device=device)
        else:
            mae = load_mae_by_name(args.embedding_model, repo_root=_REPO_ROOT, device=device)
        mae.eval()
        emb_d = int(mae.embedding_dim)
        hist_n = int(model.history_n)

        if args.pgn is not None:
            if not args.pgn.is_file():
                print(f"Error: not found {args.pgn}", file=sys.stderr)
                return 1
            agg = eval_pgn(
                args.pgn,
                model,
                mae,
                device,
                history_n=hist_n,
                embedding_dim=emb_d,
                chunk_k=args.chunk_k,
                limit_games=args.limit,
                agg_random=agg_random,
                rng=rng,
            )
        else:
            assert args.fen_file is not None
            if not args.fen_file.is_file():
                print(f"Error: not found {args.fen_file}", file=sys.stderr)
                return 1
            agg = eval_fen_file(
                args.fen_file,
                model,
                mae,
                device,
                history_n=hist_n,
                embedding_dim=emb_d,
                chunk_k=args.chunk_k,
                limit=args.limit,
                agg_random=agg_random,
                rng=rng,
            )

    if agg_random is not None:
        agg_random.skipped_not_legal = agg.skipped_not_legal

    agg.report(heading="=== Trained model ===", file=sys.stdout)
    if agg_random is not None:
        print(file=sys.stdout)
        agg_random.report(
            heading="=== Random baseline (i.i.d. N(0,1) score per legal move) ===",
            file=sys.stdout,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
