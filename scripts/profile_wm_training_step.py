#!/usr/bin/env python3
"""Time a few world_model training steps (wm_beta-shaped by default) + PyTorch profiler table.

Uses synthetic batches so HDF5 is optional. Example:

  python scripts/profile_wm_training_step.py --spec world_model/model_configs/wm_beta.yaml --stage 1 --steps 12 --warmup 3
"""

from __future__ import annotations

import argparse
import random
import sys
from contextlib import nullcontext
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from torch.profiler import ProfilerActivity, profile

from jepa2.config import BOARD_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH
from world_model.architectures import build_model
from world_model.model_spec import load_model_spec, resolve_training_config_for_stage
from world_model.training_loop import _forward_batch


def _synthetic_batch(
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    b = batch_size
    board_t = torch.zeros(b, BOARD_HEIGHT, BOARD_WIDTH, BOARD_CHANNELS, device=device)
    board_post = torch.zeros(b, BOARD_HEIGHT, BOARD_WIDTH, BOARD_CHANNELS, device=device)
    # Single legal from/to per row for masks + labels
    fs = torch.randint(0, 64, (b,), device=device, dtype=torch.long)
    ts = torch.randint(0, 64, (b,), device=device, dtype=torch.long)
    from_mask = torch.zeros(b, 64, device=device)
    to_mask = torch.zeros(b, 64, device=device)
    from_mask.scatter_(1, fs.unsqueeze(1), 1.0)
    to_mask.scatter_(1, ts.unsqueeze(1), 1.0)
    elo = torch.randn(b, device=device, dtype=torch.float32) * 200.0 + 1500.0
    pr = torch.zeros(b, device=device, dtype=torch.long)
    return board_t, board_post, from_mask, to_mask, elo, fs, ts, pr


def main() -> int:
    p = argparse.ArgumentParser(description="Profile world_model _forward_batch + backward.")
    p.add_argument("--spec", type=str, default="world_model/model_configs/wm_beta.yaml")
    p.add_argument("--stage", type=int, default=1, help="1-based stage index for resolved config")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--device", type=str, default=None, help="cuda | cpu (default: auto)")
    args = p.parse_args()

    spec_path = Path(args.spec)
    if not spec_path.is_file():
        print(f"Missing spec {spec_path}", file=sys.stderr)
        return 1

    stage_idx = int(args.stage) - 1
    if stage_idx < 0:
        print("--stage must be >= 1", file=sys.stderr)
        return 1

    spec = load_model_spec(spec_path.resolve())
    if stage_idx >= len(spec["stages"]):
        print(f"--stage {args.stage} out of range", file=sys.stderr)
        return 1

    resolved = resolve_training_config_for_stage(spec, stage_idx)
    arch = spec["architecture"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    use_amp = bool(resolved.get("use_amp", True)) and device.type == "cuda"
    bs = int(resolved["train"]["batch_size"])

    model = build_model(arch["id"], arch.get("config") or {}).to(device)
    model.train()
    if hasattr(model, "init_target_from_online"):
        model.init_target_from_online()

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4, weight_decay=0.01)

    print("resolved:", spec_path.name, f"stage={args.stage}", f"device={device}", f"batch_size={bs}", f"use_amp={use_amp}")
    print(
        "loss weights:",
        f"jepa_patch={resolved['jepa_patch_weight']}",
        f"from_ce={resolved['from_sq_ce_weight']}",
        f"recon_piece={resolved.get('recon_piece_ce_weight')}",
        f"recon_turn={resolved.get('recon_turn_ce_weight')}",
        f"recon_can_move={resolved.get('recon_can_move_ce_weight')}",
        file=sys.stderr,
    )

    def one_step(i: int, profile_fwd_bwd: bool) -> None:
        rng = random.Random(12345 + i)
        tup = _synthetic_batch(batch_size=bs, device=device)
        board_t, board_post, from_mask, to_mask, elo, fs, ts, pr = tup
        opt.zero_grad(set_to_none=True)
        if profile_fwd_bwd:
            ctx = profile(
                activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else []),
                record_shapes=False,
                with_stack=False,
            )
        else:
            ctx = nullcontext()

        with ctx as prof:
            loss, _metrics = _forward_batch(
                model,
                board_t,
                board_post,
                from_mask,
                to_mask,
                elo,
                fs,
                ts,
                pr,
                resolved,
                train=True,
                rng=rng,
                device=device,
                use_amp=use_amp,
            )
            loss.backward()
        if profile_fwd_bwd and prof is not None:
            print(prof.key_averages().table(sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total", row_limit=25))

    warmup = max(int(args.warmup), 0)
    steps = max(int(args.steps), 1)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Warmup (no profiler print)
    for i in range(warmup):
        one_step(i, profile_fwd_bwd=False)
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Timed loop
    starter = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    ender = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    import time

    if device.type == "cuda":
        total_ms = 0.0
        for i in range(steps):
            opt.zero_grad(set_to_none=True)
            rng = random.Random(99999 + i)
            board_t, board_post, from_mask, to_mask, elo, fs, ts, pr = _synthetic_batch(batch_size=bs, device=device)
            starter.record()
            loss, _ = _forward_batch(
                model,
                board_t,
                board_post,
                from_mask,
                to_mask,
                elo,
                fs,
                ts,
                pr,
                resolved,
                train=True,
                rng=rng,
                device=device,
                use_amp=use_amp,
            )
            loss.backward()
            ender.record()
            torch.cuda.synchronize()
            total_ms += starter.elapsed_time(ender)
        print(f"\nMean forward+backward per step (cuda events, {steps} steps): {total_ms / steps:.3f} ms")
    else:
        t0 = time.perf_counter()
        for i in range(steps):
            opt.zero_grad(set_to_none=True)
            rng = random.Random(99999 + i)
            board_t, board_post, from_mask, to_mask, elo, fs, ts, pr = _synthetic_batch(batch_size=bs, device=device)
            loss, _ = _forward_batch(
                model,
                board_t,
                board_post,
                from_mask,
                to_mask,
                elo,
                fs,
                ts,
                pr,
                resolved,
                train=True,
                rng=rng,
                device=device,
                use_amp=use_amp,
            )
            loss.backward()
        elapsed = time.perf_counter() - t0
        print(f"\nMean forward+backward per step (cpu wall, {steps} steps): {1000.0 * elapsed / steps:.3f} ms")

    print("\n--- Profiler: one forward+backward after warmup ---")
    one_step(10_000, profile_fwd_bwd=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
