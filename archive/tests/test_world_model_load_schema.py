"""world_model checkpoint schema: required architecture fields."""

from __future__ import annotations

import pytest

from world_model.load import world_model_architecture_fields_from_checkpoint


def test_world_model_architecture_fields_requires_id_and_config() -> None:
    with pytest.raises(KeyError, match="architecture_id"):
        world_model_architecture_fields_from_checkpoint({"model_state_dict": {}})
    with pytest.raises(KeyError, match="architecture_config"):
        world_model_architecture_fields_from_checkpoint(
            {"architecture_id": "chess_world_model_v1", "model_state_dict": {}}
        )
    aid, cfg = world_model_architecture_fields_from_checkpoint(
        {
            "architecture_id": "chess_world_model_v1",
            "architecture_config": {"d_model": 64},
            "model_state_dict": {},
        }
    )
    assert aid == "chess_world_model_v1"
    assert cfg == {"d_model": 64}


def test_world_model_architecture_id_must_be_non_empty_string() -> None:
    with pytest.raises(KeyError):
        world_model_architecture_fields_from_checkpoint(
            {"architecture_id": "", "architecture_config": {}, "model_state_dict": {}}
        )
    with pytest.raises(KeyError):
        world_model_architecture_fields_from_checkpoint(
            {"architecture_id": "   ", "architecture_config": {}, "model_state_dict": {}}
        )
