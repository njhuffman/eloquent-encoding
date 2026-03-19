# Chess board embedding (MAE)

Self-supervised chess board embedding using a **Masked AutoEncoder (MAE)**: encode masked board states to a 128-d vector; decoder reconstructs only piece positions (8×8×12) on masked squares. Loss is applied only on masked regions so the model learns from inpainting.

## Data pipeline

1. **PGN → HDF5**  
   Run from the **repository root**:

   ```bash
   python -m embedding.scripts.pgn_to_hdf5 path/to/games.pgn -o path/to/output_dir
   ```

   - Reads the PGN and assigns each game to train / val / test (80 / 10 / 10) with a fixed seed.
   - For each game, replays moves and samples positions (opening skipped; first 10 moves 90% skip; remaining 50% skip).
   - For each sampled position saves the **board tensor** (8×8×18) and **metadata** (Elo, piece counts, outcome, in-check).
   - Writes separate files: `train.h5`, `val.h5`, `test.h5` in the output directory (or next to the PGN if `-o` is omitted).
   - Progress is shown with tqdm. Use `--seed` to change the RNG seed.

2. **Training**  
   From the **repository root**:

   ```bash
   python -m embedding.train --train-h5 path/to/output_dir/train.h5 --val-h5 path/to/output_dir/val.h5
   ```

   Optional: `--batch-size`, `--epochs`, `--lr`, `--checkpoint-dir`, `--workers`.  
   Best model (by validation loss) is saved as `checkpoints/best.pt` (or the dir you pass).

3. **Linear probes (validate embedding)**  
   After training, run probes on the same train/val/test splits using only a **small subset** of the data (default 10%):

   ```bash
   python -m embedding.scripts.run_probes --train-h5 path/to/train.h5 --val-h5 path/to/val.h5 --test-h5 path/to/test.h5
   ```

   Probes: **piece count** (single-layer linear MLP, MSE), **in_check** (single-layer linear MLP, BCE), **elo regression** (linear MLP, MSE), **elo top vs bottom** (linear MLP, BCE; top N% vs bottom N% by mean Elo). Use `--subset-ratio` (default 0.1) and `--elo-quantile` (default 0.25). Checkpoint defaults to `checkpoints/best.pt`.

4. **Full pipeline (train + probes + report)**  
   Single script that trains the MAE, runs probes, and writes a report:

   ```bash
   python -m embedding.run_full_pipeline --train-h5 path/to/train.h5 --val-h5 path/to/val.h5 --test-h5 path/to/test.h5 --report embedding_report.md
   ```

   The report includes: MAE training and validation loss per epoch, final MAE test loss, and for each probe the train/val/test loss (MSE for regression probes, log loss for classification). Optional: `--epochs`, `--batch-size`, `--subset-ratio`, `--checkpoint-dir`, etc.

**Performance (GPU utilization)**  
By default, train/val boards are **loaded into RAM once** before training (~4.6 GB per 1M samples), so the DataLoader does no file I/O and the GPU stays fed. If you hit out-of-memory, use `--no-in-memory` to load from HDF5 on each access (slower). For better GPU utilization use `--workers 2`; if you see a Bus error (out of shared memory), increase `/dev/shm` (e.g. Docker: `--shm-size=256m`). Optional: `--compile` uses `torch.compile(model)` for faster forward/backward (first epoch is slower due to compilation). Use `--profile` to print a profiler summary after the first epoch.

## HDF5 schema

Each split file (`train.h5`, `val.h5`, `test.h5`) contains:

| Dataset | Shape | Dtype   | Description |
|---------|--------|---------|-------------|
| `board` | (N, 8, 8, 18) | float32 | Board tensor per position (see below). |
| `meta`  | (N, 6)         | float32 | Per-row: `[elo_white, elo_black, piece_count_white, piece_count_black, outcome, in_check]`. `outcome`: -1 = black win, 0 = draw, 1 = white win. `in_check`: 0 or 1. |

**Board tensor (8×8×18)**  
- Planes 0–5: white P, N, B, R, Q, K (one-hot per square).  
- Planes 6–11: black P, N, B, R, Q, K.  
- Plane 12: side to move (1 = white, 0 = black).  
- Planes 13–16: castling rights (white K, white Q, black K, black Q), full 8×8 layer 0 or 1.  
- Plane 17: en passant target square (1 on that square, 0 elsewhere).

## Model and masking

- **Encoder input**: 8×8×19 = board with **masked positions zeroed** (18 ch) + **mask channel** (1 ch; 1 = masked, 0 = visible). So the model cannot see piece content at masked squares.
- **Masking**: For each sample, a random fraction of squares in **[5%, 50%]** is chosen and those positions are zeroed in the board and marked in the mask channel.
- **Decoder**: Embedding (128-d) + mask (8×8×1) → 8×8×12 **piece planes only** (no turn, castling, or en passant). Loss (MSE) is applied **only on masked positions**.

## Config

All tunable constants (split ratios, skip probs, mask range, embedding dim, batch size, LR, epochs, etc.) live in **`embedding/config.py`**. Edit that file to change behavior without digging through scripts.

## Dependencies

- **Already in repo** `requirements.txt`: `numpy`, `python-chess`, `h5py`, `tqdm`, `torch`, `scikit-learn`.
- **GPU**: Training uses CUDA automatically when available (faster). For a GPU build of PyTorch with CUDA, install from [pytorch.org](https://pytorch.org) (e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu121` for CUDA 12.1). On GPU, mixed-precision (AMP) is enabled by default; use `--no-amp` to disable. Use `--device cuda:0` to pick a specific GPU. To sanity-check GPU training, run `python -m embedding.scripts.test_train_gpu` (runs a few steps on GPU if available, else CPU).

## CPU inference benchmark (encoder architecture)

To compare encoder architectures by **CPU inference time** (e.g. to pick a model for low-latency deployment), run:

```bash
pip install -r embedding/scripts/requirements-benchmark.txt
python -m embedding.scripts.benchmark_encoder_onnx
```

This builds several encoder variants (different conv depth, channel widths, embedding dim, MLP size), exports each to ONNX with random weights, and reports mean inference time on CPU for batch sizes 1, 5, and 10. Edit the `architectures` list in the script to add or change variants.

## File layout

- `config.py` — constants (split ratios, mask range, batch size, LR, etc.).
- `board_encoding.py` — `board_to_tensor()` (8×8×18), `get_piece_mask_8x8x12()`.
- `scripts/pgn_to_hdf5.py` — PGN → HDF5 with train/val/test splits and batched writes.
- `dataset.py` — PyTorch `ChessBoardDataset` (random 5–50% mask, zero + mask channel → 8×8×19).
- `model.py` — CNN encoder (8×8×19 → 128-d) and decoder (→ 8×8×12); `masked_mse_loss`.
- `train.py` — Training loop and best-model checkpointing.
- `scripts/run_probes.py` — Linear probes (piece count, in_check, elo regression, elo top vs bottom) on a subset of train/val/test.
- `run_full_pipeline.py` — End-to-end: train MAE, compute test loss, run probes, write report (training/val loss per epoch, final test loss, probe train/val/test loss).
- `scripts/benchmark_encoder_onnx.py` — ONNX CPU benchmark for encoder architecture variants (batch 1, 5, 10).
