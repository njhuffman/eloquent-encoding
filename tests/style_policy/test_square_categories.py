import torch
from style_policy.square_categories import NUM_SQUARE_CATEGORIES, square_categories_from_board_tensor
from style_policy.packed_codec import packed_to_board_tensor
import h5py


def test_num_categories_is_18():
    assert NUM_SQUARE_CATEGORIES == 18


def test_categories_shape_and_range():
    with h5py.File("/mnt/eloquence_bulk/databases/j3_training_1M.h5", "r") as f:
        packed = torch.from_numpy(f["packed_pre"][0:8].astype("uint8"))
    board = packed_to_board_tensor(packed)
    cats = square_categories_from_board_tensor(board)
    assert cats.shape == (8, 64)
    assert cats.min() >= 0 and cats.max() < NUM_SQUARE_CATEGORIES
