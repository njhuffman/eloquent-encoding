#!/usr/bin/env python3
"""Export the policy model to three ONNX graphs (encode, from_head, to_head), fp32."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import chess
from style_policy.model import BasePolicy
from style_policy.onnx_export import build_export_modules
from style_policy.board_encode import board_to_packed
from style_policy.packed_codec import packed_to_board_tensor

OPSET = 17


def board_tensor_for_fen(fen: str) -> np.ndarray:
    packed = board_to_packed(chess.Board(fen))
    return packed_to_board_tensor(packed).numpy().astype(np.float32)  # (1,8,8,18)


def export_fp32(checkpoint_path: str, out_dir) -> dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ck = torch.load(checkpoint_path, map_location="cpu")
    policy = BasePolicy.from_config(ck["architecture"])
    policy.load_state_dict(ck["model"], strict=False)  # old policy-only checkpoints predate the value head
    enc, fh, th = build_export_modules(policy)
    d = int(ck["architecture"]["d_model"])

    bt = torch.from_numpy(board_tensor_for_fen(chess.STARTING_FEN))
    with torch.no_grad():
        squares = enc(bt)
    elo = torch.tensor([15], dtype=torch.long)
    fsq = torch.tensor([0], dtype=torch.long)

    torch.onnx.export(enc, (bt,), str(out_dir / "encode.onnx"), opset_version=OPSET,
                      input_names=["board_tensor"], output_names=["squares"])
    torch.onnx.export(fh, (squares, elo), str(out_dir / "from_head.onnx"), opset_version=OPSET,
                      input_names=["squares", "elo_idx"], output_names=["from_logits"])
    torch.onnx.export(th, (squares, fsq, elo), str(out_dir / "to_head.onnx"), opset_version=OPSET,
                      input_names=["squares", "from_sq", "elo_idx"], output_names=["to_logits"])
    return {"d_model": d, "n_elo_buckets": int(ck["architecture"]["n_elo_buckets"])}


def quantize_and_check(fp32_dir, out_dir) -> dict:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    fp32_dir = Path(fp32_dir); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for name in ("encode", "from_head", "to_head"):
        dst = out_dir / f"{name}_int8.onnx"
        quantize_dynamic(str(fp32_dir / f"{name}.onnx"), str(dst), weight_type=QuantType.QInt8)
        total += dst.stat().st_size
    return {"size_bytes": total}


def main() -> int:
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="style_policy_checkpoints/base_64M/base_64M_stage_1.pt")
    ap.add_argument("--out", default="build/onnx")
    args = ap.parse_args()
    meta = export_fp32(args.checkpoint, args.out)
    print("exported fp32 ->", args.out, meta)
    fp32 = Path(args.out)
    web_pub = Path("web/public"); web_pub.mkdir(parents=True, exist_ok=True)
    info = quantize_and_check(fp32, web_pub)
    (web_pub / "model_meta.json").write_text(json.dumps(
        {"d_model": meta["d_model"], "n_elo_buckets": meta["n_elo_buckets"],
         "files": ["encode_int8.onnx", "from_head_int8.onnx", "to_head_int8.onnx"]}))
    print("int8 deploy artifacts ->", web_pub, info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
