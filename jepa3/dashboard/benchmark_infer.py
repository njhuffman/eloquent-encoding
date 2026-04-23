"""Forward-pass benchmark for jepa3 dashboard (online encode + from/to logits)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from jepa2.config import BOARD_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH
from jepa3.architectures import build_model, resolve_config_for_id
from jepa3.load import load_jepa3_from_checkpoint


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


def _jepa3_forward_step(
    model: torch.nn.Module,
    *,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> None:
    b = int(batch_size)
    board = torch.randn(b, BOARD_HEIGHT, BOARD_WIDTH, BOARD_CHANNELS, device=device, dtype=torch.float32)
    fs = torch.randint(0, 64, (b,), device=device, dtype=torch.long)
    ts = torch.randint(0, 64, (b,), device=device, dtype=torch.long)

    if use_amp:
        with torch.amp.autocast("cuda"):
            z_glob, _z_hat = model.encode_online_with_jepa(board, fs, ts)
            _ = model.forward_from_logits(z_glob)
            _ = model.forward_to_logits(z_glob, fs)
    else:
        z_glob, _z_hat = model.encode_online_with_jepa(board, fs, ts)
        _ = model.forward_from_logits(z_glob)
        _ = model.forward_to_logits(z_glob, fs)


def benchmark_forward_pass(
    spec: dict[str, Any],
    *,
    checkpoint_path: Path | None,
    batch_size: int = 8,
    warmup: int = 3,
    repeats: int = 20,
    device: torch.device | None = None,
) -> dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arch_id = spec["architecture"]["id"]
    arch_cfg = spec["architecture"].get("config") or {}

    if checkpoint_path is not None and checkpoint_path.is_file():
        model = load_jepa3_from_checkpoint(checkpoint_path, device=device)
        ckpt_mtime = checkpoint_path.stat().st_mtime
    else:
        model = build_model(arch_id, arch_cfg).to(device)
        ckpt_mtime = None

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    use_amp = device.type == "cuda"
    with torch.no_grad():
        for _ in range(warmup):
            _jepa3_forward_step(model, batch_size=batch_size, device=device, use_amp=use_amp)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times: list[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _jepa3_forward_step(model, batch_size=batch_size, device=device, use_amp=use_amp)
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
