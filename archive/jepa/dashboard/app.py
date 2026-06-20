"""FastAPI app: REST catalog, curves, benchmark, and SSE for training runs + GPU polling."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from jepa.checkpoint_paths import stage_checkpoint_path
from jepa.dashboard.benchmark_infer import benchmark_forward_pass, cache_key_for_benchmark, count_parameters_for_spec
from jepa.dashboard.orchestration import is_safe_model_name, next_missing_stage, spec_file_stem_exists
from jepa.dashboard.scan import (
    list_models_catalog,
    load_spec_by_model_name,
    read_epoch_metrics_jsonl,
    read_model_profile,
    read_stage_benchmarks,
    repo_root,
    sparse_stage_points_from_checkpoints,
    summarize_checkpoint,
    stage_grid_for_spec,
)
from jepa.metrics_paths import (
    epoch_metrics_jsonl_path,
    model_profile_json_path,
    stage_benchmarks_json_path,
)

_STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="JEPA Training Dashboard", version="1.0.0")

if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


_benchmark_cache: dict[str, dict[str, Any]] = {}


class RunBroadcaster:
    def __init__(self) -> None:
        self._subs: list[asyncio.Queue[dict[str, Any]]] = []

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=2000)
        self._subs.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        if q in self._subs:
            self._subs.remove(q)

    async def publish(self, msg: dict[str, Any]) -> None:
        for q in list(self._subs):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass


class TrainingRunManager:
    """Single active ``jepa.train`` subprocess."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._proc: asyncio.subprocess.Process | None = None
        self._gpu_task: asyncio.Task[None] | None = None
        self._read_tasks: list[asyncio.Task[None]] = []
        self.model: str | None = None
        self.stage: int | None = None
        self.broadcaster = RunBroadcaster()

    @property
    def busy(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def start(self, model: str) -> dict[str, Any]:
        if not is_safe_model_name(model) or not spec_file_stem_exists(model):
            raise ValueError("invalid model name")
        spec = load_spec_by_model_name(model)
        if spec["name"] != model:
            raise ValueError("spec name mismatch")
        stage, msg = next_missing_stage(spec)
        if stage is None:
            raise ValueError(msg)

        async with self._lock:
            if self.busy:
                raise RuntimeError("A training run is already active")
            root = repo_root()
            cmd = [
                sys.executable,
                "-m",
                "jepa.train",
                "--model",
                model,
                "--stage",
                str(stage),
            ]
            self._proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self.model = model
            self.stage = stage
            await self.broadcaster.publish(
                {
                    "type": "started",
                    "model": model,
                    "stage": stage,
                    "message": msg,
                    "cmd": cmd,
                }
            )
            self._read_tasks = [
                asyncio.create_task(self._pump_stream(self._proc.stdout, "stdout")),
                asyncio.create_task(self._pump_stream(self._proc.stderr, "stderr")),
            ]
            self._gpu_task = asyncio.create_task(self._gpu_loop())
            asyncio.create_task(self._supervise())
        return {"model": model, "stage": stage, "message": msg}

    async def _supervise(self) -> None:
        proc = self._proc
        if proc is None:
            return
        code = await proc.wait()
        for t in self._read_tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
        self._read_tasks = []
        if self._gpu_task:
            self._gpu_task.cancel()
            try:
                await self._gpu_task
            except asyncio.CancelledError:
                pass
            self._gpu_task = None
        await self.broadcaster.publish({"type": "finished", "returncode": code})
        self._proc = None
        self.model = None
        self.stage = None

    async def _pump_stream(self, stream: asyncio.StreamReader | None, name: str) -> None:
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                break
            text = line.decode(errors="replace").rstrip("\n")
            await self.broadcaster.publish({"type": "log", "stream": name, "line": text})

    async def _gpu_loop(self) -> None:
        while self.busy:
            sample = await _query_nvidia_smi_async()
            await self.broadcaster.publish({"type": "gpu", "sample": sample})
            await asyncio.sleep(1.0)
        await self.broadcaster.publish({"type": "gpu", "sample": None})


_run_manager = TrainingRunManager()


async def _query_nvidia_smi_async() -> dict[str, Any] | None:
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        if proc.returncode != 0:
            return None
        line = out.decode().strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            return None
        return {
            "utilization_gpu_pct": int(parts[0]) if parts[0].isdigit() else None,
            "memory_used_mib": int(parts[1]) if parts[1].isdigit() else None,
            "memory_total_mib": int(parts[2]) if parts[2].isdigit() else None,
        }
    except (FileNotFoundError, asyncio.TimeoutError, IndexError, ValueError):
        return None


@app.get("/favicon.ico")
async def favicon() -> Response:
    return Response(status_code=204)


@app.get("/")
async def index() -> FileResponse:
    index_path = _STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(404, "Dashboard static files missing")
    return FileResponse(index_path)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models")
async def api_models() -> list[dict[str, Any]]:
    return list_models_catalog()


@app.get("/api/models/{name}")
async def api_model_detail(name: str) -> dict[str, Any]:
    if not is_safe_model_name(name):
        raise HTTPException(400, "invalid model name")
    try:
        spec = load_spec_by_model_name(name)
    except FileNotFoundError:
        raise HTTPException(404, "model not found")
    ckpt_dir = Path(spec["checkpoint_dir"])
    grid = stage_grid_for_spec(spec)
    summaries: dict[str, Any] = {}
    for s in grid:
        if s.exists:
            summaries[str(s.stage)] = summarize_checkpoint(Path(s.checkpoint_path))
    stage_n, next_msg = next_missing_stage(spec)
    profile = read_model_profile(spec)
    if profile is not None and profile.get("n_parameters") is not None:
        n_params = int(profile["n_parameters"])
    else:
        n_params = count_parameters_for_spec(spec)
    cpu_1 = profile.get("cpu_single_forward_seconds") if profile else None
    return {
        "spec": {
            "name": spec["name"],
            "architecture": spec["architecture"],
            "checkpoint_dir": spec["checkpoint_dir"],
            "n_training_stages": len(spec["stages"]),
        },
        "n_parameters": n_params,
        "cpu_single_forward_seconds": cpu_1,
        "profile_saved": profile is not None,
        "profile_path": str(model_profile_json_path(ckpt_dir, name)),
        "stages": [
            {
                "stage": s.stage,
                "exists": s.exists,
                "blocked_by": s.blocked_by,
                "checkpoint_summary": summaries.get(str(s.stage)),
            }
            for s in grid
        ],
        "next_stage": stage_n,
        "next_stage_message": next_msg,
    }


@app.get("/api/models/{name}/curves")
async def api_curves(name: str, stage: int) -> dict[str, Any]:
    if not is_safe_model_name(name):
        raise HTTPException(400, "invalid model name")
    if stage < 1:
        raise HTTPException(400, "stage must be >= 1 for training curves")
    try:
        spec = load_spec_by_model_name(name)
    except FileNotFoundError:
        raise HTTPException(404, "model not found")
    if stage > len(spec["stages"]):
        raise HTTPException(400, "stage out of range")
    ckpt_dir = Path(spec["checkpoint_dir"])
    mpath = epoch_metrics_jsonl_path(ckpt_dir, name, stage)
    epochs = read_epoch_metrics_jsonl(mpath)
    if epochs:
        sparse: list[dict[str, Any]] = []
    else:
        sparse = [p for p in sparse_stage_points_from_checkpoints(spec) if p.get("stage") == stage]
    return {
        "model": name,
        "stage": stage,
        "epochs": epochs,
        "metrics_path": str(mpath),
        "sparse_fallback": sparse,
    }


@app.get("/api/models/{name}/stage-benchmarks")
async def api_stage_benchmarks(name: str) -> dict[str, Any]:
    if not is_safe_model_name(name):
        raise HTTPException(400, "invalid model name")
    try:
        spec = load_spec_by_model_name(name)
    except FileNotFoundError:
        raise HTTPException(404, "model not found")
    data = read_stage_benchmarks(spec)
    if data is None:
        return {
            "model": name,
            "stages": [],
            "meta": {},
            "path": str(stage_benchmarks_json_path(Path(spec["checkpoint_dir"]), name)),
            "missing": True,
        }
    return {
        "model": name,
        "stages": data.get("stages") or [],
        "meta": {k: v for k, v in data.items() if k not in ("stages", "model", "version")},
        "path": str(stage_benchmarks_json_path(Path(spec["checkpoint_dir"]), name)),
        "missing": False,
    }


@app.get("/api/models/{name}/benchmark")
async def api_benchmark(
    name: str,
    checkpoint_stage: int | None = None,
    batch_size: int = 8,
) -> dict[str, Any]:
    if not is_safe_model_name(name):
        raise HTTPException(400, "invalid model name")
    try:
        spec = load_spec_by_model_name(name)
    except FileNotFoundError:
        raise HTTPException(404, "model not found")
    ckpt_dir = Path(spec["checkpoint_dir"])
    ckpt_path: Path | None = None
    if checkpoint_stage is not None:
        if checkpoint_stage < 0 or checkpoint_stage > len(spec["stages"]):
            raise HTTPException(400, "checkpoint_stage out of range")
        p = stage_checkpoint_path(ckpt_dir, name, checkpoint_stage)
        if p.is_file():
            ckpt_path = p
    else:
        for s in range(len(spec["stages"]), -1, -1):
            p = stage_checkpoint_path(ckpt_dir, name, s)
            if p.is_file():
                ckpt_path = p
                break
    key = cache_key_for_benchmark(spec, ckpt_path) + f"|b{batch_size}"
    if key in _benchmark_cache:
        cached = dict(_benchmark_cache[key])
        cached["cached"] = True
        return cached
    result = benchmark_forward_pass(spec, checkpoint_path=ckpt_path, batch_size=batch_size)
    result["cached"] = False
    _benchmark_cache[key] = result
    return result


class StartRunBody(BaseModel):
    model: str


@app.post("/api/runs/start")
async def api_run_start(body: StartRunBody) -> dict[str, Any]:
    try:
        return await _run_manager.start(body.model)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e


@app.get("/api/runs/status")
async def api_run_status() -> dict[str, Any]:
    return {
        "busy": _run_manager.busy,
        "model": _run_manager.model,
        "stage": _run_manager.stage,
    }


_SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@app.get("/api/runs/events")
async def api_run_events() -> StreamingResponse:
    q = _run_manager.broadcaster.subscribe()

    async def gen():
        # Firefox and some proxies close the socket if no bytes are sent before the first await
        # on an empty queue; SSE comments are ignored by EventSource but flush the response.
        yield ": sse-ok\n\n"
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=25.0)
                    yield f"data: {json.dumps(msg)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            _run_manager.broadcaster.unsubscribe(q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
