import torch
from style_policy.model import BasePolicy
from style_policy.onnx_export import build_export_modules

CFG = dict(d_model=256, n_layers=8, nhead=8, dim_feedforward=1024, dropout=0.0,
           head_hidden=512, elo_dim=32, n_elo_buckets=40)

def _board_tensor(b=2):
    # random but valid: one-hot piece planes on a few squares, plane 12 turn bit
    t = torch.zeros(b, 8, 8, 18)
    t[:, 0, 0, 3] = 1.0   # a rook on a1
    t[:, 7, 4, 11] = 1.0  # a black king-ish
    t[:, :, :, 12] = 1.0  # white to move
    return t

def test_export_wrappers_match_eager():
    policy = BasePolicy.from_config(CFG).eval()
    enc, fh, th = build_export_modules(policy)
    bt = _board_tensor()
    with torch.no_grad():
        _, squares_ref = policy.encoder(bt)
        squares = enc(bt)
        assert torch.allclose(squares, squares_ref, atol=1e-5)
        elo = torch.tensor([12, 18], dtype=torch.long)
        assert torch.allclose(fh(squares, elo), policy.from_head(squares, elo_idx=elo), atol=1e-5)
        fsq = torch.tensor([0, 4], dtype=torch.long)
        assert torch.allclose(th(squares, fsq, elo), policy.to_head(squares, fsq, elo_idx=elo), atol=1e-5)


import numpy as np, onnxruntime as ort, chess
from pathlib import Path
from scripts.export_onnx import export_fp32, board_tensor_for_fen
from style_policy.model import BasePolicy
from style_policy.model_spec import elo_to_bucket

CKPT = "style_policy_checkpoints/base_64M/base_64M_stage_1.pt"
FENS = [chess.STARTING_FEN,
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1"]

def test_fp32_onnx_parity(tmp_path):
    ck = torch.load(CKPT, map_location="cpu")
    policy = BasePolicy.from_config(ck["architecture"]); policy.load_state_dict(ck["model"]); policy.eval()
    export_fp32(CKPT, tmp_path)
    enc = ort.InferenceSession(str(tmp_path / "encode.onnx"))
    fh = ort.InferenceSession(str(tmp_path / "from_head.onnx"))
    th = ort.InferenceSession(str(tmp_path / "to_head.onnx"))
    for fen in FENS:
        bt = board_tensor_for_fen(fen)
        elo = np.array([15], dtype=np.int64)
        with torch.no_grad():
            _, sq_ref = policy.encoder(torch.from_numpy(bt))
            fl_ref = policy.from_head(sq_ref, elo_idx=torch.from_numpy(elo)).numpy()
        sq = enc.run(None, {"board_tensor": bt})[0]
        assert np.allclose(sq, sq_ref.numpy(), atol=1e-4)
        fl = fh.run(None, {"squares": sq, "elo_idx": elo})[0]
        assert np.allclose(fl, fl_ref, atol=1e-4)
        fsq = np.array([int(fl_ref.argmax())], dtype=np.int64)
        tl = th.run(None, {"squares": sq, "from_sq": fsq, "elo_idx": elo})[0]
        tl_ref = policy.to_head(sq_ref, torch.from_numpy(fsq), elo_idx=torch.from_numpy(elo)).detach().numpy()
        assert np.allclose(tl, tl_ref, atol=1e-4)


def test_int8_parity_top1(tmp_path):
    from scripts.export_onnx import export_fp32, quantize_and_check, board_tensor_for_fen
    ck = torch.load(CKPT, map_location="cpu")
    policy = BasePolicy.from_config(ck["architecture"]); policy.load_state_dict(ck["model"]); policy.eval()
    export_fp32(CKPT, tmp_path / "fp32")
    quantize_and_check(tmp_path / "fp32", tmp_path / "int8")
    enc = ort.InferenceSession(str(tmp_path / "int8" / "encode_int8.onnx"))
    fh = ort.InferenceSession(str(tmp_path / "int8" / "from_head_int8.onnx"))
    for fen in FENS:
        bt = board_tensor_for_fen(fen); elo = np.array([15], dtype=np.int64)
        with torch.no_grad():
            _, sq_ref = policy.encoder(torch.from_numpy(bt))
            fl_ref = policy.from_head(sq_ref, elo_idx=torch.from_numpy(elo)).numpy()
        sq = enc.run(None, {"board_tensor": bt})[0]
        fl = fh.run(None, {"squares": sq, "elo_idx": elo})[0]
        assert int(fl.argmax()) == int(fl_ref.argmax())          # top-1 from-square preserved
        assert np.allclose(fl, fl_ref, atol=0.15)


def test_fixtures_written(tmp_path, monkeypatch):
    from scripts.gen_web_fixtures import build_cases
    cases = build_cases(CKPT, FENS, elo=1500)
    assert len(cases["cases"]) == len(FENS)
    c = cases["cases"][0]
    assert len(c["board_tensor"]) == 8 * 8 * 18
    assert len(c["from_logits"]) == 64 and len(c["legal_from"]) == 64
    assert c["bucket"] == 15
