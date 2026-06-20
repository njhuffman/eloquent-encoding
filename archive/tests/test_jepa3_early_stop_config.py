"""early_stop_joint_top1 in resolved jepa3 training config."""

from __future__ import annotations

import pytest

from jepa3.model_spec import MODEL_CONFIGS_DIR, load_model_spec, resolve_training_config_for_stage


def test_early_stop_joint_top1_default_none() -> None:
    spec = load_model_spec(MODEL_CONFIGS_DIR / "tiny_test.yaml")
    r = resolve_training_config_for_stage(spec, 0)
    assert r.get("early_stop_joint_top1") is None


def test_early_stop_joint_top1_stage_override() -> None:
    spec = load_model_spec(MODEL_CONFIGS_DIR / "tiny_test.yaml")
    spec["stages"][0]["early_stop_joint_top1"] = 0.5
    r = resolve_training_config_for_stage(spec, 0)
    assert r["early_stop_joint_top1"] == 0.5


def test_early_stop_joint_top1_invalid() -> None:
    spec = load_model_spec(MODEL_CONFIGS_DIR / "tiny_test.yaml")
    spec["stages"][0]["early_stop_joint_top1"] = 1.01
    with pytest.raises(ValueError, match="early_stop_joint_top1"):
        resolve_training_config_for_stage(spec, 0)
