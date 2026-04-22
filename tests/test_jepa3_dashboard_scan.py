"""Smoke tests for jepa3 dashboard catalog (no server)."""

from __future__ import annotations

from jepa3.dashboard.scan import iter_model_spec_paths, list_models_catalog, load_spec_by_model_name, repo_root


def test_repo_root_is_eloquence() -> None:
    r = repo_root()
    assert (r / "jepa3").is_dir()


def test_iter_model_spec_paths_includes_tiny_test() -> None:
    stems = {p.stem for p in iter_model_spec_paths()}
    assert "tiny_test" in stems


def test_list_models_catalog_contains_tiny_test() -> None:
    rows = list_models_catalog()
    names = {r["name"] for r in rows}
    assert "tiny_test" in names


def test_load_spec_tiny_test() -> None:
    spec = load_spec_by_model_name("tiny_test")
    assert spec["name"] == "tiny_test"
    assert spec["architecture"]["id"] == "chess_jepa_v3"
