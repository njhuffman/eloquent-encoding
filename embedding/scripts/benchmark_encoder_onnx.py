#!/usr/bin/env python3
"""
Benchmark embedding encoder architectures on CPU via ONNX.

Varies: number of conv layers, channel widths, embedding dim, MLP depth.
Builds encoder-only models with random weights, exports to ONNX, and times
inference with onnxruntime (CPU) for batch sizes 1, 5, and 10.

Usage:
  pip install onnx onnxruntime
  python -m embedding.scripts.benchmark_encoder_onnx
"""

from __future__ import annotations

import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

# Add repo root for embedding package
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import onnx  # noqa: F401
except ImportError:
    print("Install onnx: pip install onnx", file=sys.stderr)
    sys.exit(1)
try:
    import onnxruntime as ort
except ImportError:
    print("Install onnxruntime: pip install onnxruntime", file=sys.stderr)
    sys.exit(1)

from embedding.config import ENCODER_INPUT_CHANNELS

# Fixed input spatial size (chess board)
H = 8
W = 8


@dataclass
class EncoderArch:
    """Encoder architecture parameters."""

    name: str
    # Each element: (out_channels, stride). Stride 2 halves spatial size.
    conv_stages: list[tuple[int, int]]
    embedding_dim: int
    # Hidden sizes for MLP head (before final linear to embedding_dim). Can be empty.
    mlp_hidden: list[int] = field(default_factory=list)

    def num_conv_layers(self) -> int:
        return len(self.conv_stages)


def _conv_out_size(size: int, stride: int, kernel: int = 3, padding: int = 1) -> int:
    return (size + 2 * padding - kernel) // stride + 1


def build_encoder(arch: EncoderArch) -> nn.Module:
    """Build a PyTorch encoder (B, C_in, H, W) -> (B, embedding_dim)."""
    in_ch = ENCODER_INPUT_CHANNELS
    layers: list[nn.Module] = []
    h, w = H, W
    for out_ch, stride in arch.conv_stages:
        layers.append(nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        in_ch = out_ch
        h = _conv_out_size(h, stride)
        w = _conv_out_size(w, stride)
    layers.append(nn.Flatten())
    flat_size = in_ch * h * w
    for hidden in arch.mlp_hidden:
        layers.append(nn.Linear(flat_size, hidden))
        layers.append(nn.ReLU(inplace=True))
        flat_size = hidden
    layers.append(nn.Linear(flat_size, arch.embedding_dim))
    return nn.Sequential(*layers)


def export_encoder_onnx(encoder: nn.Module, path: Path, batch_size: int = 1) -> None:
    """Export encoder to ONNX with fixed batch size for cleaner graphs."""
    encoder.eval()
    dummy = torch.randn(batch_size, ENCODER_INPUT_CHANNELS, H, W)
    torch.onnx.export(
        encoder,
        dummy,
        str(path),
        input_names=["board"],
        output_names=["embedding"],
        dynamic_axes={
            "board": {0: "batch"},
            "embedding": {0: "batch"},
        },
        opset_version=14,
        do_constant_folding=True,
    )


def count_parameters(module: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in module.parameters())


def _format_size(size_bytes: int) -> str:
    """Human-readable file size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def run_onnx_timing(onnx_path: Path, batch_sizes: list[int], num_warmup: int = 5, num_repeat: int = 50) -> dict[int, float]:
    """Run ONNX model on CPU for given batch sizes; return mean time per forward (seconds)."""
    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
        sess_options=ort.SessionOptions(),
    )
    name_in = sess.get_inputs()[0].name
    results: dict[int, float] = {}
    for bs in batch_sizes:
        x = torch.randn(bs, ENCODER_INPUT_CHANNELS, H, W).numpy()
        # Warmup
        for _ in range(num_warmup):
            sess.run(None, {name_in: x})
        # Timed runs
        start = time.perf_counter()
        for _ in range(num_repeat):
            sess.run(None, {name_in: x})
        elapsed = time.perf_counter() - start
        results[bs] = elapsed / num_repeat
    return results


def main() -> int:
    batch_sizes = [1, 5, 10]
    num_warmup = 10
    num_repeat = 100

    # Architecture variants: name -> EncoderArch
    # Conv stages: (out_channels, stride). Two stride-2 steps take 8x8 -> 2x2.
    architectures = [
        EncoderArch(
            name="tiny_64d",
            conv_stages=[(32, 1), (32, 2), (64, 2)],
            embedding_dim=64,
            mlp_hidden=[128],
        ),
        EncoderArch(
            name="small_128d",
            conv_stages=[(64, 1), (64, 2), (128, 2)],
            embedding_dim=128,
            mlp_hidden=[256],
        ),
        EncoderArch(
            name="base_128d",
            conv_stages=[(64, 1), (64, 1), (128, 2), (256, 2)],
            embedding_dim=128,
            mlp_hidden=[512],
        ),
        EncoderArch(
            name="base_256d",
            conv_stages=[(64, 1), (64, 1), (128, 2), (256, 2)],
            embedding_dim=256,
            mlp_hidden=[512],
        ),
        EncoderArch(
            name="deep_128d",
            conv_stages=[
                (64, 1),
                (64, 1),
                (64, 1),
                (128, 2),
                (128, 1),
                (256, 2),
            ],
            embedding_dim=128,
            mlp_hidden=[512, 512],
        ),
        EncoderArch(
            name="wide_128d",
            conv_stages=[(96, 1), (96, 2), (192, 2)],
            embedding_dim=128,
            mlp_hidden=[512],
        ),
        EncoderArch(
            name="minimal_64d",
            conv_stages=[(32, 2), (64, 2)],
            embedding_dim=64,
            mlp_hidden=[],
        ),
    ]

    print("Encoder ONNX CPU benchmark (inference time per forward pass)")
    print("Warmup runs:", num_warmup, "| Timed runs:", num_repeat)
    print("Batch sizes:", batch_sizes)
    print()

    with tempfile.TemporaryDirectory(prefix="embed_onnx_bench_") as tmpdir:
        tmp = Path(tmpdir)
        for arch in architectures:
            encoder = build_encoder(arch)
            encoder.apply(lambda m: _init_weights(m) if isinstance(m, (nn.Conv2d, nn.Linear)) else None)
            encoder.eval()
            onnx_path = tmp / f"{arch.name}.onnx"
            export_encoder_onnx(encoder, onnx_path, batch_size=1)
            num_params = count_parameters(encoder)
            onnx_size_bytes = onnx_path.stat().st_size
            timings = run_onnx_timing(onnx_path, batch_sizes, num_warmup=num_warmup, num_repeat=num_repeat)
            print(f"  {arch.name}")
            print(f"    parameters={num_params:,}  compiled_size={_format_size(onnx_size_bytes)}")
            print(f"    conv_layers={arch.num_conv_layers()} embedding_dim={arch.embedding_dim} mlp_hidden={arch.mlp_hidden}")
            for bs in batch_sizes:
                t_ms = timings[bs] * 1000
                print(f"    batch={bs}: {t_ms:.2f} ms")
            print()
    return 0


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)


if __name__ == "__main__":
    raise SystemExit(main())
