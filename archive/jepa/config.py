"""
Chess-JEPA defaults: data generation, HDF5 layout, training, artifacts.
Board geometry matches embedding (reuse embedding.config constants).
"""

from embedding.config import BOARD_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH

# Re-export for jepa modules
__all__ = [
    "BOARD_CHANNELS",
    "BOARD_HEIGHT",
    "BOARD_WIDTH",
    "DEFAULT_ARCHITECTURE_ID",
    "NUM_NEGATIVES_DEFAULT",
    "HDF5_FLUSH_BATCH_SIZE",
    "TRAIN_RATIO",
    "VAL_RATIO",
    "TEST_RATIO",
    "OPENING_SKIP_HALF_MOVES",
    "FIRST_10_MOVES_SKIP_PROB",
    "REMAINING_MOVES_SKIP_PROB",
    "BATCH_SIZE",
    "LEARNING_RATE",
    "WEIGHT_DECAY",
    "NUM_EPOCHS",
    "DATALOADER_NUM_WORKERS",
    "CHECKPOINT_DIR",
    "BEST_CHECKPOINT_NAME",
    "LOG_INTERVAL",
    "ARTIFACTS_DIR",
    "REGISTRY_FILENAME",
    "CHECKPOINT_BASENAME",
    "MODEL_CONFIGS_DIR",
    "EMA_MOMENTUM_DEFAULT",
    "TRIPLET_MARGIN_ALPHA_DEFAULT",
    "VICREG_VAR_COEF_DEFAULT",
    "VICREG_STD_TARGET",
    "PROBE_SUBSET_RATIO",
    "ELO_QUANTILE",
    "PROBE_RANDOM_SEED",
    "PROBE_EPOCHS",
    "PROBE_LR",
]

# --- PGN → JEPA HDF5 (mirrors embedding/config.py sampling knobs) ---
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
OPENING_SKIP_HALF_MOVES = 0
FIRST_10_MOVES_SKIP_PROB = 0.9
REMAINING_MOVES_SKIP_PROB = 0.5
HDF5_FLUSH_BATCH_SIZE = 50_000
NUM_NEGATIVES_DEFAULT = 8

# --- Training ---
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
NUM_EPOCHS = 50
DATALOADER_NUM_WORKERS = 4
EMA_MOMENTUM_DEFAULT = 0.999
TRIPLET_MARGIN_ALPHA_DEFAULT = 0.2
VICREG_VAR_COEF_DEFAULT = 0.1
VICREG_STD_TARGET = 1.0

CHECKPOINT_DIR = "jepa_checkpoints"
BEST_CHECKPOINT_NAME = "best.pt"
LOG_INTERVAL = 100

ARTIFACTS_DIR = "jepa/artifacts"
REGISTRY_FILENAME = "registry.json"
CHECKPOINT_BASENAME = "checkpoint.pt"
MODEL_CONFIGS_DIR = "jepa/model_configs"

# Default architecture id (see jepa.architectures)
DEFAULT_ARCHITECTURE_ID = "chess_jepa_v1"

# --- Linear probes (jepa/scripts/run_probes.py); align with embedding/config.py ---
PROBE_SUBSET_RATIO = 0.1
ELO_QUANTILE = 0.25
PROBE_RANDOM_SEED = 42
PROBE_EPOCHS = 100
PROBE_LR = 1e-2
