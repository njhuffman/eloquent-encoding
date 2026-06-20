"""Import jepa3 FastAPI app (requires fastapi in env)."""

from __future__ import annotations

import pytest

try:
    from jepa3.dashboard import app as app_mod
except ImportError as e:  # pragma: no cover
    pytest.skip(f"jepa3.dashboard import failed: {e}", allow_module_level=True)


def test_app_has_routes() -> None:
    paths = {r.path for r in app_mod.app.routes}
    assert "/api/health" in paths
    assert "/api/models" in paths
