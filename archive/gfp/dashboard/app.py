"""FastAPI app for gfp: model catalog and per-stage metrics (read-only)."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from gfp.model_spec import load_model_spec, spec_path_for_model
from gfp.dashboard.scan import (
    is_safe_model_name,
    list_model_names,
    metrics_path_for_stage,
    read_stage_metrics_json,
    stage_grid_for_spec,
    stage_status_as_dict,
)

_STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="gfp Training Dashboard", version="1.0.0")

if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    index_path = _STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=500, detail="static/index.html missing")
    return FileResponse(index_path, media_type="text/html")


@app.get("/api/models")
def api_models() -> dict:
    return {"models": list_model_names()}


@app.get("/api/models/{name}/summary")
def api_model_summary(name: str) -> dict:
    if not is_safe_model_name(name):
        raise HTTPException(status_code=404, detail="invalid model name")
    try:
        spec_path = spec_path_for_model(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="model spec not found") from None
    try:
        spec = load_model_spec(spec_path)
    except (OSError, ValueError, TypeError, KeyError) as e:
        raise HTTPException(status_code=400, detail=f"invalid spec: {e}") from e
    if spec["name"] != name:
        raise HTTPException(status_code=400, detail="spec name mismatch")

    grid = stage_grid_for_spec(spec)
    stages_out: list[dict] = []
    for st in grid:
        row = stage_status_as_dict(st)
        metrics = None
        if st.stage >= 1:
            mp = metrics_path_for_stage(spec, st.stage)
            metrics = read_stage_metrics_json(mp)
        row["metrics"] = metrics
        stages_out.append(row)

    return {
        "name": spec["name"],
        "checkpoint_dir": spec["checkpoint_dir"],
        "encoder_checkpoint": spec["encoder_checkpoint"],
        "train_dataset_h5": spec["train_dataset_h5"],
        "val_dataset_h5": spec["val_dataset_h5"],
        "architecture": spec["architecture"],
        "n_stages": len(spec["stages"]),
        "stages": stages_out,
    }
