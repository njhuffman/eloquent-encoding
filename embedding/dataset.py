"""
PyTorch Dataset for chess board HDF5: loads boards and applies random masking (5--50%)
with zeroed masked positions and mask channel for encoder input (8x8x19).
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .board_encoding import get_piece_mask_8x8x12
from .config import (
    BOARD_CHANNELS,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    ENCODER_INPUT_CHANNELS,
    MAX_MASK_RATIO,
    MIN_MASK_RATIO,
    PIECE_PLANES,
)


class ChessBoardDataset(Dataset):
    """
    HDF5 dataset of board tensors (N, 8, 8, 18). Each __getitem__ returns:
    - encoder_input: (8, 8, 19) float32 — board with masked positions zeroed + mask channel (1 = masked).
    - mask: (8, 8, 1) float32 — 1.0 where masked, 0.0 where visible.
    - target_piece: (8, 8, 12) float32 — piece planes for loss (only applied on masked positions).
    """

    def __init__(self, h5_path: str | Path, seed: int | None = None):
        self.h5_path = Path(h5_path)
        self._seed = seed
        with open(self.h5_path, "rb") as _:
            pass  # check readable
        import h5py
        with h5py.File(self.h5_path, "r") as f:
            self._len = f["board"].shape[0]

    def __len__(self) -> int:
        return self._len

    def _random_mask(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (mask_8x8, indices_of_masked). mask_8x8 is 1.0 where masked, 0.0 visible."""
        n_squares = BOARD_HEIGHT * BOARD_WIDTH
        # Random fraction in [MIN_MASK_RATIO, MAX_MASK_RATIO]
        frac = random.uniform(MIN_MASK_RATIO, MAX_MASK_RATIO)
        n_masked = max(1, min(n_squares - 1, int(round(frac * n_squares))))
        indices = random.sample(range(n_squares), n_masked)
        mask_flat = np.zeros(n_squares, dtype=np.float32)
        mask_flat[indices] = 1.0
        mask_8x8 = mask_flat.reshape(BOARD_HEIGHT, BOARD_WIDTH)
        return mask_8x8, set(indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        import h5py
        if self._seed is not None:
            # Deterministic per-index for reproducibility in val/test
            rng = random.Random(self._seed + idx)
            n_squares = BOARD_HEIGHT * BOARD_WIDTH
            frac = rng.uniform(MIN_MASK_RATIO, MAX_MASK_RATIO)
            n_masked = max(1, min(n_squares - 1, int(round(frac * n_squares))))
            indices = set(rng.sample(range(n_squares), n_masked))
            mask_8x8 = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
            for i in indices:
                r, c = i // BOARD_WIDTH, i % BOARD_WIDTH
                mask_8x8[r, c] = 1.0
        else:
            mask_8x8, _ = self._random_mask()

        with h5py.File(self.h5_path, "r") as f:
            board = f["board"][idx]  # (8, 8, 18)

        board = np.asarray(board, dtype=np.float32)
        # Zero out masked positions (all 18 channels at those squares)
        masked_board = board.copy()
        masked_board[mask_8x8 == 1.0, :] = 0.0
        # Encoder input: zeroed board (18 ch) + mask channel (1 ch) = 19 ch
        mask_channel = mask_8x8[:, :, np.newaxis]  # (8, 8, 1)
        encoder_input = np.concatenate([masked_board, mask_channel], axis=-1)  # (8, 8, 19)
        target_piece = get_piece_mask_8x8x12(board)  # (8, 8, 12)

        return (
            torch.from_numpy(encoder_input),
            torch.from_numpy(mask_channel),
            torch.from_numpy(target_piece),
        )


def get_dataloaders(
    train_h5: str | Path,
    val_h5: str | Path,
    batch_size: int,
    num_workers: int = 0,
    val_seed: int = 0,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build train and val DataLoaders. Train is shuffled; val uses fixed seed for masking."""
    from torch.utils.data import DataLoader
    train_ds = ChessBoardDataset(train_h5, seed=None)
    val_ds = ChessBoardDataset(val_h5, seed=val_seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
