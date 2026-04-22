"""Tests for jepa3 board tensor -> probe targets."""

from __future__ import annotations

import torch

from jepa3.architectures.board_probe_labels import meta_targets_from_board, piece_labels_64_from_board


def test_piece_labels_one_hot_round_trip() -> None:
    b = torch.zeros(1, 8, 8, 18)
    b[0, 3, 5, 7] = 1.0
    lab = piece_labels_64_from_board(b)
    idx = 3 * 8 + 5
    assert lab.shape == (1, 64)
    assert int(lab[0, idx].item()) == 7
    assert int(lab[0, 0].item()) == 12


def test_piece_labels_empty_board() -> None:
    b = torch.zeros(2, 8, 8, 18)
    lab = piece_labels_64_from_board(b)
    assert (lab == 12).all()


def test_meta_turn_castle_ep() -> None:
    b = torch.zeros(1, 8, 8, 18)
    b[0, 0, 0, 12] = 1.0
    b[0, 0, 0, 13] = 1.0
    b[0, 0, 0, 15] = 1.0
    b[0, 2, 3, 17] = 1.0
    m = meta_targets_from_board(b)
    assert m["turn"].shape == (1,)
    assert float(m["turn"][0].item()) == 1.0
    assert m["castle"].shape == (1, 4)
    assert float(m["castle"][0, 0].item()) == 1.0
    assert float(m["castle"][0, 1].item()) == 0.0
    assert float(m["castle"][0, 2].item()) == 1.0
    assert float(m["castle"][0, 3].item()) == 0.0
    ep_sq = 2 * 8 + 3
    assert int(m["ep_class"][0].item()) == 1 + ep_sq


def test_meta_no_ep_class_zero() -> None:
    b = torch.zeros(1, 8, 8, 18)
    m = meta_targets_from_board(b)
    assert int(m["ep_class"][0].item()) == 0
