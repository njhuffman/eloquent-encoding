# Move predictor

Two GRUs over the last **N** frozen MAE board embeddings per color (white-to-move positions vs black-to-move positions), concatenated with **side to move first**, plus a small **turn embedding**; an MLP scores each candidate as `concat(board_emb, from_emb, to_emb, style_pair, turn_emb)`. Training is 3-way cross-entropy (chosen vs two negatives).

## Data layout (HDF5)

- `cur_emb`: MAE embedding of the current position (before the predicted move).
- `hist_white_emb`, `hist_black_emb` (float32, `(S, N, D)`): last up to **N** prior board embeddings where it was white / black to move in that snapshot (chronological within each stream, **right-padded** with zeros).
- `hist_white_len`, `hist_black_len` (int32): valid prefix lengths per stream.
- `side_to_move` (uint8): `0` = white to move, `1` = black to move (must match `fen`).
- `from_sq`, `to_sq` (uint8, shape `(S,3)`): square indices 0–63; columns are shuffled per row.
- `label` (uint8): index 0–2 of the played move.
- `promotion` (uint8, `(S,3)`): `0` = none, else python-chess `piece_type` of promoted piece (2–5). Stored for future training; the current model ignores it.
- `fen`: position before the move (for hard-negative mining).

**Regenerate** HDF5 after this layout change; older files with `hist_emb` / `hist_len` only are not read by the current code.

## 1. Build stage-1 HDF5 from PGN

Requires a trained MAE checkpoint or a registered name in `embedding/artifacts/registry.json`:

```bash
python -m move_predictor.scripts.pgn_to_move_hdf5 games.pgn -o data/move_h5 \
  --embedding-model YOUR_MAE_NAME \
  --history-n 8
# or:
python -m move_predictor.scripts.pgn_to_move_hdf5 games.pgn -o data/move_h5 \
  --checkpoint path/to/best.pt \
  --history-n 8
```

Options: `--game-skip-prob`, `--move-skip-prob`, `--seed`, `--max-samples`, `--flush-size`, `--device`.

Outputs `train.h5`, `val.h5`, `test.h5` (80/10/10 split) when any samples land in each split.

## 2. Train stage-1

Training uses **`MovePredictorH5Dataset`** with **`BatchSampler`**: each worker opens the HDF5 once (`swmr=True`) and **`__getitems__`** loads a whole batch by merging **contiguous row slices** after sorting indices (shuffle-safe). No full-file RAM load. Tune throughput mainly with **`--batch-size`** and **`--workers`**.

```bash
python -m move_predictor.train \
  --train-h5 data/move_h5/train.h5 \
  --val-h5 data/move_h5/val.h5 \
  --checkpoint-dir move_predictor_checkpoints
```

Each epoch logs **`[pipeline]`** timings on stderr: mean **wait on DataLoader** (CPU blocked until the batch is ready), **host→device** copy, and **forward + backward + optimizer step**. Batch 0 is skipped in the mean (cold start). With CUDA, the GPU segment is wall time after launching work unless you pass **`--profile-sync-gpu`**, which calls `torch.cuda.synchronize()` every batch so that segment reflects true device wait (slower). Disable the lines with **`--no-pipeline-log`**.

`non_blocking` copies when CUDA is available; workers use `persistent_workers` and `prefetch_factor=2` when `--workers` > 0.

If reads fail with `OSError` / errno 5 on a stripe, the file may be corrupt at that offset—see `python -m move_predictor.scripts.verify_move_h5` and regenerate or copy the HDF5.

## 3. Mine hard negatives (stage-2 data)

```bash
python -m move_predictor.scripts.mine_hard_negatives \
  --input-h5 data/move_h5/train.h5 \
  --checkpoint move_predictor_checkpoints/best.pt \
  -o data/move_h5/train_hard.h5
```

Rows with fewer than two legal alternatives to the labeled move are skipped.

## 4. Train stage-2

Point `--train-h5` / `--val-h5` at mined HDF5 files (and use a fresh `--checkpoint-dir` if you want a separate stage-2 checkpoint).

## 5. Full legal-set evaluation

Training accuracy is 3-way among sampled negatives. To measure how often the model ranks the **played** move among **all** legal moves:

```bash
python -m move_predictor.evaluate --checkpoint move_predictor_checkpoints/best.pt \
  --test-h5 data/move_h5/test.h5
```

Uses stored embeddings, dual histories, `side_to_move`, and reconstructs the played move from `label` + `from_sq` / `to_sq` / `promotion`. Prints top-1 / top-3 / top-5 / top-10 hit rate, mean rank, MRR, and mean branching factor.

**PGN** (replays games, rebuilds per-color history like the HDF5 builder; needs the same MAE used for embeddings):

```bash
python -m move_predictor.evaluate --checkpoint .../best.pt --pgn games.pgn \
  --mae-checkpoint path/to/mae_best.pt
```

**FEN + UCI file** (one position per line: `FEN<TAB>UCI` or `FEN|UCI`, or a single line `…full fen… uci`; both color histories empty):

```bash
python -m move_predictor.evaluate --checkpoint .../best.pt --fen-file positions.txt \
  --embedding-model YOUR_MAE_NAME
```

Options: `--limit`, `--chunk-k` (legal moves per forward pass), `--device`.

By default a **random baseline** is printed too: each legal move gets an independent standard-normal score and the same rank/top-k metrics (so top-1 rate is near **1 / mean branching factor** on average). Disable with `--no-baseline-random`; use `--random-seed` for reproducibility.

## Reference

Board tensors and MAE loading match [embedding/board_encoding.py](../embedding/board_encoding.py) and [embedding/load.py](../embedding/load.py). Encoder inputs use a zero mask channel as in [embedding/scripts/run_probes.py](../embedding/scripts/run_probes.py).
