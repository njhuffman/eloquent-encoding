"""Defaults for move predictor data generation and training."""

from pathlib import Path

# --- PGN sampling (mirrors embedding/config.py spirit) ---
GAME_SKIP_PROB = 0.0
MOVE_SKIP_PROB = 0.9
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

HDF5_FLUSH_BATCH_SIZE = 10_000

# --- Model ---
HISTORY_N = 8
MOVE_EMB_DIM = 16
TURN_EMB_DIM = 4
GRU_HIDDEN = 8
GRU_NUM_LAYERS = 1
MLP_HIDDEN = 32
DROPOUT = 0.1

# --- Training ---
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
DATALOADER_NUM_WORKERS = 4
CHECKPOINT_DIR = Path("move_predictor_checkpoints")
