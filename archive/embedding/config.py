"""
Configuration for chess board embedding: data collection, HDF5 schema, and training.
Tweak these in one place instead of scattering magic numbers.
"""

# --- Data collection (pgn_to_hdf5) ---
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Skip this many half-moves from the start of each game (opening)
OPENING_SKIP_HALF_MOVES = 0

# After opening: for the first 10 half-moves, skip each with this probability
FIRST_10_MOVES_SKIP_PROB = 0.9

# For remaining half-moves, skip with this probability
REMAINING_MOVES_SKIP_PROB = 0.5

# Write to HDF5 every this many boards (per split) to limit memory
HDF5_FLUSH_BATCH_SIZE = 50_000

# --- Board encoding ---
# Board tensor shape (8, 8, 18): 6 white piece planes + 6 black + turn + 4 castling + 1 en passant
BOARD_HEIGHT = 8
BOARD_WIDTH = 8
BOARD_CHANNELS = 18
PIECE_PLANES = 12  # first 12 channels are piece positions (decoder target)

# --- Training (masking and model) ---
# Mask fraction per sample: sample uniformly in [MIN_MASK_RATIO, MAX_MASK_RATIO]
MIN_MASK_RATIO = 0.49
MAX_MASK_RATIO = 0.51

EMBEDDING_DIM = 128

# Encoder input: zeroed board (18 ch) + mask channel (1 ch) = 19 channels
ENCODER_INPUT_CHANNELS = BOARD_CHANNELS + 1

# --- Training loop ---
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
DATALOADER_NUM_WORKERS = 4

# Checkpoint and logging
CHECKPOINT_DIR = "checkpoints"
BEST_CHECKPOINT_NAME = "best.pt"
LOG_INTERVAL = 100  # print loss every N batches

# Registered trained models (see embedding.registry, embedding.load)
ARTIFACTS_DIR = "embedding/artifacts"
REGISTRY_FILENAME = "registry.json"
CHECKPOINT_BASENAME = "checkpoint.pt"

# Training specs (JSON): see embedding/model_configs/ and embedding.training_spec
MODEL_CONFIGS_DIR = "embedding/model_configs"

# --- Linear probes (validate_embedding) ---
# Use only this fraction of train/val/test for probe training and eval (small subset)
PROBE_SUBSET_RATIO = 0.1
# For elo top/bottom probe: use top N% vs bottom N% (e.g. 0.25 = 25%)
ELO_QUANTILE = 0.25
PROBE_RANDOM_SEED = 42
# Single-layer linear MLP probe training
PROBE_EPOCHS = 100
PROBE_LR = 1e-2
