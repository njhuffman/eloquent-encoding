"""Parameter counts and a small forward-pass latency benchmark for dashboard metrics."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from jepa.architectures import build_model, resolve_config_for_id
from jepa.config import BOARD_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH
from jepa.load import load_jepa_from_checkpoint


def count_parameters_for_spec(spec: dict[str, Any]) -> int:
    arch_id = spec["architecture"]["id"]
    arch_cfg = spec["architecture"].get("config") or {}
    m = build_model(arch_id, arch_cfg)
    return sum(p.numel() for p in m.parameters())


def _arch_fingerprint(spec: dict[str, Any]) -> str:
    arch_id = spec["architecture"]["id"]
    resolved = resolve_config_for_id(arch_id, spec["architecture"].get("config") or {})
    raw = json.dumps({"id": arch_id, "cfg": resolved}, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def benchmark_forward_pass(
    spec: dict[str, Any],
    *,
    checkpoint_path: Path | None,
    batch_size: int = 8,
    warmup: int = 3,
    repeats: int = 20,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Warmup + timed ``forward_online`` batches. Uses checkpoint weights if path exists and is a file,
    else random initialization from architecture.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arch_id = spec["architecture"]["id"]
    arch_cfg = spec["architecture"].get("config") or {}

    if checkpoint_path is not None and checkpoint_path.is_file():
        model = load_jepa_from_checkpoint(checkpoint_path, device=device)
        ckpt_mtime = checkpoint_path.stat().st_mtime
    else:
        model = build_model(arch_id, arch_cfg).to(device)
        ckpt_mtime = None

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    board = torch.randn(
        batch_size, BOARD_HEIGHT, BOARD_WIDTH, BOARD_CHANNELS, device=device, dtype=torch.float32
    )
    elo = torch.randn(batch_size, device=device, dtype=torch.float32) * 1500.0

    use_amp = device.type == "cuda"
    with torch.no_grad():
        for _ in range(warmup):
            if use_amp:
                with torch.amp.autocast("cuda"):
                    model.forward_online(board, elo)
            else:
                model.forward_online(board, elo)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times: list[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    model.forward_online(board, elo)
            else:
                model.forward_online(board, elo)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    mean_s = sum(times) / max(len(times), 1)
    return {
        "n_parameters": n_params,
        "batch_size": batch_size,
        "device": str(device),
        "use_amp": use_amp,
        "warmup": warmup,
        "repeats": repeats,
        "mean_batch_seconds": mean_s,
        "mean_sample_seconds": mean_s / batch_size,
        "checkpoint_used": checkpoint_path is not None and checkpoint_path.is_file(),
        "checkpoint_mtime": ckpt_mtime,
        "arch_fingerprint": _arch_fingerprint(spec),
    }


def cache_key_for_benchmark(spec: dict[str, Any], checkpoint_path: Path | None) -> str:
    fp = _arch_fingerprint(spec)
    m = ""
    if checkpoint_path is not None and checkpoint_path.is_file():
        m = str(int(checkpoint_path.stat().st_mtime))
    return f"{spec['name']}|{fp}|{m}"
