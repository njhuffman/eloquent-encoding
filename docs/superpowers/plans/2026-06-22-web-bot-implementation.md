# Web Bot (GitHub Pages, client-side) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `PolicyBot` move predictor playable in a browser on GitHub Pages — fully client-side (no backend), with elo selection and a "what the model is thinking" panel.

**Architecture:** Export the 6.8M-param PyTorch model to three ONNX graphs (`encode`, `from_head`, `to_head`), int8-quantized to ~7MB. A framework-agnostic TypeScript inference core runs them in the browser via `onnxruntime-web`, doing legality masking / temperature / sampling in JS (mirroring `style_policy/play.py`). `chess.js` provides rules and legal moves; `react-chessboard` renders the board; a React side panel shows the policy's top moves and a from/to probability heatmap. Deployed as a static Vite build via GitHub Actions.

**Tech Stack:** Python (PyTorch, onnx, onnxruntime) for export + parity; TypeScript, React 18, Vite, Vitest, chess.js, react-chessboard, onnxruntime-web (browser) / onnxruntime-node (parity tests).

## Global Constraints

- **Deploy model:** `style_policy_checkpoints/base_64M/base_64M_stage_1.pt`. Architecture is fixed: `d_model=256, n_layers=8, nhead=8, dim_feedforward=1024, dropout=0.0, head_hidden=512, elo_dim=32, n_elo_buckets=40`. Do not retrain.
- **Packed/plane layout is authoritative** and documented in `style_policy/packed_codec.py` and `style_policy/square_categories.py`. Plane order 0–11 = WP,WN,WB,WR,WQ,WK,BP,BN,BB,BR,BQ,BK; plane 12 = side-to-move (1.0 if White); planes 13–16 = castling H1,A1,H8,A8; plane 17 = en-passant target. Square index `s = rank*8 + file`, a1=0 … h8=63.
- **elo→bucket** must match `style_policy/model_spec.py::elo_to_bucket` exactly: `bucket = clamp(floor(elo/100), 0, n-1)` for `elo>0`, else `n` (the null index). With `n=40`, valid buckets 0–39, null = 40.
- **No Git LFS** (GitHub Pages serves the LFS pointer text, not the binary). The deploy artifact is committed directly: `web/public/policy_int8.onnx`. Never commit `.pt` files or the `*_checkpoints/` tree (already gitignored).
- **Parity tolerances:** fp32 ONNX vs PyTorch logits `atol=1e-4`; int8 ONNX vs PyTorch top-1 move must match on every fixture position, and logit `atol=0.15`.
- **Sampling default for the web bot:** temperature `T` is user-controlled; the inference core must support both greedy (argmax) and seeded multinomial sampling. Parity tests use greedy (`T→0`).
- Run all container commands with `OMP_NUM_THREADS=6` to avoid the CPU-oversubscription thrash documented in `docs/DEVLOG.md`.
- Web app lives under `web/`. Python export code under `scripts/` and `style_policy/`; Python tests under `tests/style_policy/`.

---

## File Structure

**Python (export + cross-language oracle):**
- `style_policy/onnx_export.py` — thin `nn.Module` wrappers (`EncodeExport`, `FromHeadExport`, `ToHeadExport`) that expose ONNX-traceable forwards starting from a board tensor (skipping the numpy `packed_to_board_tensor`).
- `scripts/export_onnx.py` — load checkpoint, export the three graphs (fp32), int8-quantize, run the parity gate, write `web/public/policy_int8.onnx` + a metadata JSON.
- `scripts/gen_web_fixtures.py` — emit `web/src/inference/__fixtures__/*.json`: board tensors, elo-bucket table, and PyTorch reference logits/top-moves for a set of FENs (the oracle the JS tests check against).
- `tests/style_policy/test_onnx_export.py` — export-wrapper equivalence + fp32/int8 parity gate.

**Web (`web/`):**
- `web/package.json`, `web/vite.config.ts`, `web/tsconfig.json`, `web/index.html`, `web/vitest.config.ts`
- `web/src/inference/boardTensor.ts` — `chess.js` Board → `Float32Array(8*8*18)`.
- `web/src/inference/elo.ts` — `eloToBucket(elo, n)`.
- `web/src/inference/legal.ts` — `legalFromMask(board)`, `legalToMask(board, fromSq)` → `boolean[64]`; square-name↔index helpers.
- `web/src/inference/sample.ts` — `maskedSoftmax`, `pickIndex` (greedy or seeded sample).
- `web/src/inference/engine.ts` — `Engine` class: loads the 3 ONNX sessions, `policy(board, elo)` → distributions, `chooseMove(board, elo, opts)` → move.
- `web/src/inference/rng.ts` — small seedable RNG (mulberry32) for reproducible sampling/tests.
- `web/src/App.tsx`, `web/src/components/BoardPanel.tsx`, `web/src/components/ThinkingPanel.tsx`, `web/src/components/Controls.tsx`
- `web/public/policy_int8.onnx` — deploy artifact (committed).
- `.github/workflows/pages.yml` — build + deploy.
- `.devcontainer/devcontainer.json` — add Node feature.

---

### Task 1: Export wrappers + ONNX/parity Python deps

**Files:**
- Create: `style_policy/onnx_export.py`
- Modify: `.devcontainer/Dockerfile` (add `onnx`, `onnxruntime`)
- Test: `tests/style_policy/test_onnx_export.py`

**Interfaces:**
- Consumes: `BasePolicy` (`style_policy/model.py`), `BoardEncoder` (`style_policy/board_encoder.py`), `FromHead`/`ToHead` (`style_policy/policy_heads.py`).
- Produces:
  - `EncodeExport(encoder)` with `forward(board_tensor: (B,8,8,18) float32) -> squares (B,64,d_model)`.
  - `FromHeadExport(from_head)` with `forward(squares: (B,64,d), elo_idx: (B,) int64) -> (B,64) float32`.
  - `ToHeadExport(to_head)` with `forward(squares: (B,64,d), from_sq: (B,) int64, elo_idx: (B,) int64) -> (B,64) float32`.
  - `build_export_modules(policy: BasePolicy) -> tuple[EncodeExport, FromHeadExport, ToHeadExport]` (sets eval, disables nested-tensor).

- [ ] **Step 1: Add deps to the dev image**

In `.devcontainer/Dockerfile`, after the `requirements.txt` install block (line 17), add:

```dockerfile
# ONNX export + runtime (web-bot tooling)
RUN pip install --no-cache-dir "onnx>=1.16" "onnxruntime>=1.18"
```

For the running container (no rebuild needed to proceed): `docker exec <container> bash -lc 'pip install "onnx>=1.16" "onnxruntime>=1.18"'`.

- [ ] **Step 2: Write the failing test**

`tests/style_policy/test_onnx_export.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `docker exec <container> bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 python -m pytest tests/style_policy/test_onnx_export.py -q'`
Expected: FAIL — `ModuleNotFoundError: No module named 'style_policy.onnx_export'`.

- [ ] **Step 4: Implement the wrappers**

`style_policy/onnx_export.py`:

```python
"""ONNX-traceable wrappers. They start from a board tensor (B,8,8,18) so the numpy
packed_to_board_tensor step is done outside the graph (in JS). The tricky square-category
logic stays *inside* the encoder graph, guaranteeing parity with training."""
from __future__ import annotations
import torch
import torch.nn as nn


class EncodeExport(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, board_tensor: torch.Tensor) -> torch.Tensor:
        _, squares = self.encoder(board_tensor)
        return squares


class FromHeadExport(nn.Module):
    def __init__(self, from_head: nn.Module):
        super().__init__()
        self.from_head = from_head

    def forward(self, squares: torch.Tensor, elo_idx: torch.Tensor) -> torch.Tensor:
        return self.from_head(squares, elo_idx=elo_idx)


class ToHeadExport(nn.Module):
    def __init__(self, to_head: nn.Module):
        super().__init__()
        self.to_head = to_head

    def forward(self, squares: torch.Tensor, from_sq: torch.Tensor, elo_idx: torch.Tensor) -> torch.Tensor:
        return self.to_head(squares, from_sq, elo_idx=elo_idx)


def build_export_modules(policy):
    policy.eval()
    # nn.TransformerEncoder's nested-tensor fast path uses data-dependent ops that don't trace.
    enc = policy.encoder
    if hasattr(enc, "encoder") and hasattr(enc.encoder, "enable_nested_tensor"):
        enc.encoder.enable_nested_tensor = False
    return (EncodeExport(policy.encoder).eval(),
            FromHeadExport(policy.from_head).eval(),
            ToHeadExport(policy.to_head).eval())
```

- [ ] **Step 5: Run test to verify it passes**

Run: `docker exec <container> bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 python -m pytest tests/style_policy/test_onnx_export.py -q'`
Expected: PASS (1 passed).

- [ ] **Step 6: Commit**

```bash
git add style_policy/onnx_export.py tests/style_policy/test_onnx_export.py .devcontainer/Dockerfile
git commit -m "feat: ONNX export wrappers for the policy model"
```

---

### Task 2: Export the three graphs (fp32) + parity gate

**Files:**
- Create: `scripts/export_onnx.py`
- Test: extend `tests/style_policy/test_onnx_export.py`

**Interfaces:**
- Consumes: `build_export_modules`, the checkpoint, `style_policy.board_encode.board_to_packed`, `style_policy.packed_codec.packed_to_board_tensor`.
- Produces: `export_fp32(checkpoint_path, out_dir) -> dict` writing `encode.onnx`, `from_head.onnx`, `to_head.onnx` into `out_dir`, returning `{"d_model":256, "n_elo_buckets":40}`.
- Produces: `board_tensor_for_fen(fen: str) -> np.ndarray (1,8,8,18) float32` (test/oracle helper) built via `chess.Board(fen)` → `board_to_packed` → `packed_to_board_tensor`.

- [ ] **Step 1: Write the failing parity test**

Append to `tests/style_policy/test_onnx_export.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec <container> bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 python -m pytest tests/style_policy/test_onnx_export.py::test_fp32_onnx_parity -q'`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.export_onnx'`.

- [ ] **Step 3: Implement the exporter**

`scripts/export_onnx.py`:

```python
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
    policy.load_state_dict(ck["model"])
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="style_policy_checkpoints/base_64M/base_64M_stage_1.pt")
    ap.add_argument("--out", default="build/onnx")
    args = ap.parse_args()
    meta = export_fp32(args.checkpoint, args.out)
    print("exported fp32 ->", args.out, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec <container> bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 python -m pytest tests/style_policy/test_onnx_export.py::test_fp32_onnx_parity -q'`
Expected: PASS. If `TransformerEncoder` export errors on the nested-tensor path, confirm Step 1/Task 1 disabled it; as a fallback add `torch.onnx.export(..., dynamo=False)`.

- [ ] **Step 5: Commit**

```bash
git add scripts/export_onnx.py tests/style_policy/test_onnx_export.py
git commit -m "feat: fp32 ONNX export of the three policy graphs with parity test"
```

---

### Task 3: int8 quantization + deploy artifact

**Files:**
- Modify: `scripts/export_onnx.py` (add `quantize_and_check`, write `web/public/policy_int8.onnx` + `web/public/model_meta.json`)
- Test: extend `tests/style_policy/test_onnx_export.py`

**Interfaces:**
- Produces: `quantize_and_check(fp32_dir, out_onnx_dir) -> dict` — dynamic-quantizes the three graphs, concatenates nothing (keeps three files but under one prefix), returns `{"size_bytes": int}`. For simplicity the three int8 graphs are written as `encode_int8.onnx`, `from_head_int8.onnx`, `to_head_int8.onnx` and the web app loads all three. `model_meta.json` records `{"d_model":256,"n_elo_buckets":40,"files":[...]}`.

> Decision note: "policy_int8.onnx" in the constraints refers to the *set* of int8 graphs shipped under `web/public/`. Three small files (~2–3MB each) are simpler than fusing into one graph and keep the export trivially correct.

- [ ] **Step 1: Write the failing int8 parity test**

Append to `tests/style_policy/test_onnx_export.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec <container> bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 python -m pytest tests/style_policy/test_onnx_export.py::test_int8_parity_top1 -q'`
Expected: FAIL — `cannot import name 'quantize_and_check'`.

- [ ] **Step 3: Implement quantization**

Add to `scripts/export_onnx.py`:

```python
def quantize_and_check(fp32_dir, out_dir) -> dict:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    fp32_dir = Path(fp32_dir); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for name in ("encode", "from_head", "to_head"):
        dst = out_dir / f"{name}_int8.onnx"
        quantize_dynamic(str(fp32_dir / f"{name}.onnx"), str(dst), weight_type=QuantType.QInt8)
        total += dst.stat().st_size
    return {"size_bytes": total}
```

And extend `main()` so the default pipeline writes the deploy artifacts:

```python
    # after export_fp32(...)
    import json, shutil
    fp32 = Path(args.out)
    web_pub = Path("web/public"); web_pub.mkdir(parents=True, exist_ok=True)
    info = quantize_and_check(fp32, web_pub)
    (web_pub / "model_meta.json").write_text(json.dumps(
        {"d_model": meta["d_model"], "n_elo_buckets": meta["n_elo_buckets"],
         "files": ["encode_int8.onnx", "from_head_int8.onnx", "to_head_int8.onnx"]}))
    print("int8 deploy artifacts ->", web_pub, info)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec <container> bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 python -m pytest tests/style_policy/test_onnx_export.py -q'`
Expected: PASS (all export tests). Then produce the real artifact:
`docker exec <container> bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 python scripts/export_onnx.py'` and confirm `web/public/*_int8.onnx` total < 10MB.

- [ ] **Step 5: Commit (artifact committed in Task 14, not here)**

```bash
git add scripts/export_onnx.py tests/style_policy/test_onnx_export.py
git commit -m "feat: int8 quantization + deploy-artifact emission with top-1 parity gate"
```

---

### Task 4: Cross-language fixtures (the JS oracle)

**Files:**
- Create: `scripts/gen_web_fixtures.py`
- Create (generated, committed): `web/src/inference/__fixtures__/cases.json`

**Interfaces:**
- Produces `cases.json`: `{"d_model":256,"n_elo_buckets":40,"cases":[{"fen","elo","board_tensor":[8*8*18 floats],"bucket","from_logits":[64],"to_from_sq","to_logits":[64],"legal_from":[64 bools],"legal_to":[64 bools],"top_move_uci"}]}`. The JS tests use this to verify `boardTensor`, `eloToBucket`, `legal*`, and the full engine path without a Python runtime.

- [ ] **Step 1: Write the failing test**

`tests/style_policy/test_onnx_export.py`, append:

```python
def test_fixtures_written(tmp_path, monkeypatch):
    from scripts.gen_web_fixtures import build_cases
    cases = build_cases(CKPT, FENS, elo=1500)
    assert len(cases["cases"]) == len(FENS)
    c = cases["cases"][0]
    assert len(c["board_tensor"]) == 8 * 8 * 18
    assert len(c["from_logits"]) == 64 and len(c["legal_from"]) == 64
    assert c["bucket"] == 15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec <container> bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 python -m pytest tests/style_policy/test_onnx_export.py::test_fixtures_written -q'`
Expected: FAIL — `No module named 'scripts.gen_web_fixtures'`.

- [ ] **Step 3: Implement the fixture generator**

`scripts/gen_web_fixtures.py`:

```python
#!/usr/bin/env python3
"""Generate JSON fixtures so the TS inference tests can check against PyTorch outputs."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch, chess
from style_policy.model import BasePolicy
from style_policy.model_spec import elo_to_bucket
from style_policy.board_encode import board_to_packed, legal_from_u64, legal_to_u64
from style_policy.packed_codec import packed_to_board_tensor

DEFAULT_FENS = [chess.STARTING_FEN,
                "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
                "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1"]


def _bits(u64: int) -> list[bool]:
    return [bool((u64 >> i) & 1) for i in range(64)]


def build_cases(checkpoint_path: str, fens: list[str], elo: int) -> dict:
    ck = torch.load(checkpoint_path, map_location="cpu")
    policy = BasePolicy.from_config(ck["architecture"]); policy.load_state_dict(ck["model"]); policy.eval()
    n = int(ck["architecture"]["n_elo_buckets"])
    bucket = int(elo_to_bucket(torch.tensor([elo]), n).item())
    cases = []
    for fen in fens:
        board = chess.Board(fen)
        bt = packed_to_board_tensor(board_to_packed(board)).float()
        with torch.no_grad():
            _, sq = policy.encoder(bt)
            fl = policy.from_head(sq, elo_idx=torch.tensor([bucket]))[0]
            from_sq = int(fl.masked_fill(~torch.tensor(_bits(legal_from_u64(board))), float("-inf")).argmax())
            tl = policy.to_head(sq, torch.tensor([from_sq]), elo_idx=torch.tensor([bucket]))[0]
            to_legal = _bits(legal_to_u64(board, from_sq))
            to_sq = int(tl.masked_fill(~torch.tensor(to_legal), float("-inf")).argmax())
        cases.append({
            "fen": fen, "elo": elo, "bucket": bucket,
            "board_tensor": bt.reshape(-1).tolist(),
            "from_logits": fl.tolist(), "legal_from": _bits(legal_from_u64(board)),
            "to_from_sq": from_sq, "to_logits": tl.tolist(), "legal_to": to_legal,
            "top_move_uci": chess.Move(from_sq, to_sq).uci(),
        })
    return {"d_model": int(ck["architecture"]["d_model"]), "n_elo_buckets": n, "cases": cases}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="style_policy_checkpoints/base_64M/base_64M_stage_1.pt")
    ap.add_argument("--out", default="web/src/inference/__fixtures__/cases.json")
    ap.add_argument("--elo", type=int, default=1500)
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_cases(args.checkpoint, DEFAULT_FENS, args.elo)))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test + generate the fixture file**

Run: `docker exec <container> bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 python -m pytest tests/style_policy/test_onnx_export.py::test_fixtures_written -q && OMP_NUM_THREADS=6 python scripts/gen_web_fixtures.py'`
Expected: PASS, then `wrote web/src/inference/__fixtures__/cases.json`.

- [ ] **Step 5: Commit**

```bash
git add scripts/gen_web_fixtures.py web/src/inference/__fixtures__/cases.json tests/style_policy/test_onnx_export.py
git commit -m "feat: cross-language fixtures for the TS inference oracle"
```

---

### Task 5: Node toolchain + web project scaffold

**Files:**
- Modify: `.devcontainer/devcontainer.json` (add Node feature)
- Create: `web/package.json`, `web/tsconfig.json`, `web/vite.config.ts`, `web/vitest.config.ts`, `web/index.html`, `web/src/main.tsx`, `web/src/App.tsx`, `web/.gitignore`

**Interfaces:**
- Produces a runnable Vite+React+TS app with Vitest. `npm run dev`, `npm run build`, `npm test` all work.

- [ ] **Step 1: Add Node to the devcontainer**

In `.devcontainer/devcontainer.json`, add (merge into existing JSON):

```json
"features": { "ghcr.io/devcontainers/features/node:1": { "version": "20" } }
```

For the running container without a rebuild, install Node 20 via nodesource:
`docker exec <container> bash -lc 'curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get install -y nodejs && node --version'`

- [ ] **Step 2: Scaffold the project files**

`web/package.json`:

```json
{
  "name": "eloquent-web-bot",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview",
    "test": "vitest run"
  },
  "dependencies": {
    "chess.js": "^1.0.0",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-chessboard": "^4.7.2",
    "onnxruntime-web": "^1.18.0"
  },
  "devDependencies": {
    "@types/react": "^18.3.3",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.1",
    "onnxruntime-node": "^1.18.0",
    "typescript": "^5.5.0",
    "vite": "^5.3.0",
    "vitest": "^2.0.0"
  }
}
```

`web/vite.config.ts` (base path is the repo name for Pages — set in Task 14; default `/` for dev):

```ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({ plugins: [react()], base: process.env.VITE_BASE ?? "/" });
```

`web/vitest.config.ts`:

```ts
import { defineConfig } from "vitest/config";
export default defineConfig({ test: { environment: "node", include: ["src/**/*.test.ts"] } });
```

`web/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2020", "lib": ["ES2020", "DOM", "DOM.Iterable"], "module": "ESNext",
    "moduleResolution": "bundler", "jsx": "react-jsx", "strict": true,
    "esModuleInterop": true, "skipLibCheck": true, "noEmit": true, "resolveJsonModule": true
  },
  "include": ["src"]
}
```

`web/index.html`:

```html
<!doctype html><html><head><meta charset="utf-8" /><title>Eloquent Bot</title></head>
<body><div id="root"></div><script type="module" src="/src/main.tsx"></script></body></html>
```

`web/src/main.tsx`:

```tsx
import React from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
createRoot(document.getElementById("root")!).render(<App />);
```

`web/src/App.tsx`:

```tsx
import React from "react";
export function App() { return <h1>Eloquent Bot</h1>; }
```

`web/.gitignore`:

```
node_modules/
dist/
```

- [ ] **Step 3: Install and verify the toolchain**

Run: `docker exec <container> bash -lc 'cd /workspaces/eloquent-encoding/web && npm install && npm run build && npm test'`
Expected: build succeeds; `npm test` reports "No test files found" (acceptable at this step) or 0 tests. (Tasks 6+ add tests.)

- [ ] **Step 4: Commit**

```bash
git add web/package.json web/package-lock.json web/tsconfig.json web/vite.config.ts web/vitest.config.ts web/index.html web/src/main.tsx web/src/App.tsx web/.gitignore .devcontainer/devcontainer.json
git commit -m "chore: scaffold web/ (Vite+React+TS+Vitest) and add Node to devcontainer"
```

---

### Task 6: `boardTensor` — chess.js Board → Float32Array(8*8*18)

**Files:**
- Create: `web/src/inference/boardTensor.ts`
- Test: `web/src/inference/boardTensor.test.ts`

**Interfaces:**
- Produces: `boardToTensor(board: Chess): Float32Array` length `8*8*18`, layout `[rank*8+file)*18 + plane]` matching `packed_to_board_tensor(...).reshape(-1)`.
- Produces: `squareToIndex(name: string): number` (`"a1"`→0 … `"h8"`→63), `indexToSquare(i: number): string`.

- [ ] **Step 1: Write the failing test**

`web/src/inference/boardTensor.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { Chess } from "chess.js";
import { boardToTensor, squareToIndex } from "./boardTensor";
import fixtures from "./__fixtures__/cases.json";

describe("boardToTensor", () => {
  it("matches python packed_to_board_tensor for every fixture", () => {
    for (const c of fixtures.cases) {
      const t = boardToTensor(new Chess(c.fen));
      expect(t.length).toBe(c.board_tensor.length);
      for (let i = 0; i < t.length; i++) expect(t[i]).toBeCloseTo(c.board_tensor[i], 5);
    }
  });
  it("indexes squares a1=0, h8=63", () => {
    expect(squareToIndex("a1")).toBe(0);
    expect(squareToIndex("h8")).toBe(63);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec <container> bash -lc 'cd /workspaces/eloquent-encoding/web && npx vitest run src/inference/boardTensor.test.ts'`
Expected: FAIL — cannot find module `./boardTensor`.

- [ ] **Step 3: Implement**

`web/src/inference/boardTensor.ts`:

```ts
import type { Chess } from "chess.js";

const C = 18;
// plane index for a piece: white p,n,b,r,q,k -> 0..5 ; black -> 6..11
const PIECE_PLANE: Record<string, number> = { p: 0, n: 1, b: 2, r: 3, q: 4, k: 5 };

export function squareToIndex(name: string): number {
  const file = name.charCodeAt(0) - 97;      // 'a' -> 0
  const rank = name.charCodeAt(1) - 49;      // '1' -> 0
  return rank * 8 + file;
}
export function indexToSquare(i: number): string {
  return String.fromCharCode(97 + (i % 8)) + String.fromCharCode(49 + Math.floor(i / 8));
}

export function boardToTensor(board: Chess): Float32Array {
  const t = new Float32Array(64 * C);
  // pieces: board.board() is rank 8..1, file a..h
  const rows = board.board();
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const piece = rows[r][f];
      if (!piece) continue;
      const sq = (7 - r) * 8 + f;            // rows[0] is rank 8 -> rank index 7
      const plane = PIECE_PLANE[piece.type] + (piece.color === "w" ? 0 : 6);
      t[sq * C + plane] = 1.0;
    }
  }
  const white = board.turn() === "w";
  for (let s = 0; s < 64; s++) if (white) t[s * C + 12] = 1.0;     // plane 12 side-to-move
  // castling: chess.js getCastlingRights
  const wc = board.getCastlingRights("w"), bc = board.getCastlingRights("b");
  const setPlane = (plane: number, on: boolean) => { if (on) for (let s = 0; s < 64; s++) t[s * C + plane] = 1.0; };
  setPlane(13, wc.k); setPlane(14, wc.q); setPlane(15, bc.k); setPlane(16, bc.q);
  // en passant
  const fen = board.fen().split(" ");
  const ep = fen[3];
  if (ep && ep !== "-") t[squareToIndex(ep) * C + 17] = 1.0;
  return t;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec <container> bash -lc 'cd /workspaces/eloquent-encoding/web && npx vitest run src/inference/boardTensor.test.ts'`
Expected: PASS. If the EP plane mismatches, note: python-chess sets `ep_square` only when a capture is legal; chess.js FEN field 4 always lists the square. The fixtures use positions without EP, so field 4 is `-`; if a future fixture has EP, align by checking `board.fen()` exactly as python-chess would. (Starting and the listed FENs have no EP.)

- [ ] **Step 5: Commit**

```bash
git add web/src/inference/boardTensor.ts web/src/inference/boardTensor.test.ts
git commit -m "feat: boardToTensor (chess.js -> 18-plane tensor) with python parity"
```

---

### Task 7: `eloToBucket`

**Files:**
- Create: `web/src/inference/elo.ts`
- Test: `web/src/inference/elo.test.ts`

**Interfaces:**
- Produces: `eloToBucket(elo: number, n: number): number` matching `model_spec.elo_to_bucket`.

- [ ] **Step 1: Write the failing test**

`web/src/inference/elo.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { eloToBucket } from "./elo";
describe("eloToBucket", () => {
  it("matches python: floor(elo/100) clamped to [0,n-1], 0 -> null index n", () => {
    expect(eloToBucket(1500, 40)).toBe(15);
    expect(eloToBucket(1200, 40)).toBe(12);
    expect(eloToBucket(50, 40)).toBe(0);
    expect(eloToBucket(9999, 40)).toBe(39);   // clamp
    expect(eloToBucket(0, 40)).toBe(40);       // null bucket
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec <container> bash -lc 'cd /workspaces/eloquent-encoding/web && npx vitest run src/inference/elo.test.ts'`
Expected: FAIL — cannot find module `./elo`.

- [ ] **Step 3: Implement**

`web/src/inference/elo.ts`:

```ts
export function eloToBucket(elo: number, n: number): number {
  if (elo > 0) return Math.min(Math.max(Math.floor(elo / 100), 0), n - 1);
  return n; // null / unknown-elo index
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec <container> bash -lc 'cd /workspaces/eloquent-encoding/web && npx vitest run src/inference/elo.test.ts'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/src/inference/elo.ts web/src/inference/elo.test.ts
git commit -m "feat: eloToBucket matching model_spec"
```

---

### Task 8: Legal-move masks from chess.js

**Files:**
- Create: `web/src/inference/legal.ts`
- Test: `web/src/inference/legal.test.ts`

**Interfaces:**
- Consumes: `squareToIndex` (from `boardTensor.ts`).
- Produces: `legalFromMask(board: Chess): boolean[]` (length 64); `legalToMask(board: Chess, fromSq: number): boolean[]` (length 64).

- [ ] **Step 1: Write the failing test**

`web/src/inference/legal.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { Chess } from "chess.js";
import { legalFromMask, legalToMask } from "./legal";
import { squareToIndex } from "./boardTensor";
import fixtures from "./__fixtures__/cases.json";

describe("legal masks", () => {
  it("matches python legal_from / legal_to per fixture", () => {
    for (const c of fixtures.cases) {
      const board = new Chess(c.fen);
      expect(legalFromMask(board)).toEqual(c.legal_from);
      expect(legalToMask(board, c.to_from_sq)).toEqual(c.legal_to);
    }
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec <container> bash -lc 'cd /workspaces/eloquent-encoding/web && npx vitest run src/inference/legal.test.ts'`
Expected: FAIL — cannot find module `./legal`.

- [ ] **Step 3: Implement**

`web/src/inference/legal.ts`:

```ts
import type { Chess } from "chess.js";
import { squareToIndex } from "./boardTensor";

export function legalFromMask(board: Chess): boolean[] {
  const m = new Array(64).fill(false);
  for (const mv of board.moves({ verbose: true })) m[squareToIndex(mv.from)] = true;
  return m;
}
export function legalToMask(board: Chess, fromSq: number): boolean[] {
  const m = new Array(64).fill(false);
  for (const mv of board.moves({ verbose: true })) {
    if (squareToIndex(mv.from) === fromSq) m[squareToIndex(mv.to)] = true;
  }
  return m;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec <container> bash -lc 'cd /workspaces/eloquent-encoding/web && npx vitest run src/inference/legal.test.ts'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/src/inference/legal.ts web/src/inference/legal.test.ts
git commit -m "feat: legal-from/legal-to masks from chess.js"
```

---

### Task 9: Masked softmax + pick (greedy / seeded sample)

**Files:**
- Create: `web/src/inference/rng.ts`, `web/src/inference/sample.ts`
- Test: `web/src/inference/sample.test.ts`

**Interfaces:**
- Produces: `mulberry32(seed: number): () => number` (uniform [0,1)).
- Produces: `maskedSoftmax(logits: Float32Array | number[], legal: boolean[], temperature: number): Float32Array` (illegal entries → 0 probability; legal entries softmaxed at `logits/temperature`).
- Produces: `pickIndex(probs: Float32Array, opts: { greedy: boolean; rand?: () => number }): number`.

- [ ] **Step 1: Write the failing test**

`web/src/inference/sample.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { maskedSoftmax, pickIndex } from "./sample";
import { mulberry32 } from "./rng";

describe("maskedSoftmax / pickIndex", () => {
  it("zeros illegal, sums to 1 over legal", () => {
    const legal = [true, false, true, false];
    const p = maskedSoftmax([2, 9, 1, 9], legal, 1.0);
    expect(p[1]).toBe(0); expect(p[3]).toBe(0);
    expect(p[0] + p[2]).toBeCloseTo(1, 6);
    expect(p[0]).toBeGreaterThan(p[2]);
  });
  it("greedy picks the max-prob legal index", () => {
    const p = maskedSoftmax([2, 9, 1, 9], [true, false, true, false], 1.0);
    expect(pickIndex(p, { greedy: true })).toBe(0);
  });
  it("seeded sampling is deterministic", () => {
    const p = maskedSoftmax([1, 1, 1, 1], [true, true, true, true], 1.0);
    const a = pickIndex(p, { greedy: false, rand: mulberry32(42) });
    const b = pickIndex(p, { greedy: false, rand: mulberry32(42) });
    expect(a).toBe(b);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec <container> bash -lc 'cd /workspaces/eloquent-encoding/web && npx vitest run src/inference/sample.test.ts'`
Expected: FAIL — cannot find module `./sample`.

- [ ] **Step 3: Implement**

`web/src/inference/rng.ts`:

```ts
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
```

`web/src/inference/sample.ts`:

```ts
export function maskedSoftmax(
  logits: Float32Array | number[], legal: boolean[], temperature: number,
): Float32Array {
  const n = logits.length;
  const t = Math.max(temperature, 1e-6);
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (legal[i] && logits[i] / t > max) max = logits[i] / t;
  const out = new Float32Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    if (!legal[i]) continue;
    const e = Math.exp(logits[i] / t - max);
    out[i] = e; sum += e;
  }
  if (sum > 0) for (let i = 0; i < n; i++) out[i] /= sum;
  return out;
}

export function pickIndex(probs: Float32Array, opts: { greedy: boolean; rand?: () => number }): number {
  if (opts.greedy) {
    let best = 0, bestv = -Infinity;
    for (let i = 0; i < probs.length; i++) if (probs[i] > bestv) { bestv = probs[i]; best = i; }
    return best;
  }
  const r = (opts.rand ?? Math.random)();
  let acc = 0;
  for (let i = 0; i < probs.length; i++) { acc += probs[i]; if (r <= acc) return i; }
  return probs.length - 1;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec <container> bash -lc 'cd /workspaces/eloquent-encoding/web && npx vitest run src/inference/sample.test.ts'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/src/inference/rng.ts web/src/inference/sample.ts web/src/inference/sample.test.ts
git commit -m "feat: masked softmax + greedy/seeded pick"
```

---

### Task 10: `Engine` — three-session two-stage inference (end-to-end parity gate)

**Files:**
- Create: `web/src/inference/engine.ts`
- Test: `web/src/inference/engine.node.test.ts` (uses `onnxruntime-node` + the committed int8 graphs)

**Interfaces:**
- Consumes: `boardToTensor`, `eloToBucket`, `legalFromMask`, `legalToMask`, `maskedSoftmax`, `pickIndex`, `indexToSquare`, `squareToIndex`.
- Produces:
  - `type Ort = { InferenceSession: { create(path: string | ArrayBuffer): Promise<any> }, Tensor: any }` (injected, so node tests pass `onnxruntime-node` and the app passes `onnxruntime-web`).
  - `class Engine` with `static load(ort, urls: {encode,fromHead,toHead}, meta: {nEloBuckets:number}): Promise<Engine>`.
  - `policy(board, elo): Promise<{ fromProbs: Float32Array; fromSq: number; toProbs: Float32Array; toSq: number }>` (greedy from/to for display).
  - `chooseMove(board, elo, opts: { temperature: number; greedy?: boolean; rand?: () => number }): Promise<{ from: string; to: string; promotion?: "q" }>`.

- [ ] **Step 1: Write the failing parity test**

`web/src/inference/engine.node.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import * as ort from "onnxruntime-node";
import { Chess } from "chess.js";
import { Engine } from "./engine";
import { indexToSquare } from "./boardTensor";
import fixtures from "./__fixtures__/cases.json";

describe("Engine (int8) parity vs python fixtures", () => {
  it("greedy top move matches python top_move_uci", async () => {
    const eng = await Engine.load(ort as any, {
      encode: "public/encode_int8.onnx",
      fromHead: "public/from_head_int8.onnx",
      toHead: "public/to_head_int8.onnx",
    }, { nEloBuckets: fixtures.n_elo_buckets });
    for (const c of fixtures.cases) {
      const board = new Chess(c.fen);
      const mv = await eng.chooseMove(board, c.elo, { temperature: 1, greedy: true });
      // python top_move_uci is from+to (+promotion); compare squares
      expect(mv.from + mv.to).toBe(c.top_move_uci.slice(0, 4));
    }
  });
});
```

(Note: the test loads from `web/public/*_int8.onnx`; run after Task 3 produced them and they were copied/committed. The `vitest.config.ts` `cwd` is `web/`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec <container> bash -lc 'cd /workspaces/eloquent-encoding/web && npx vitest run src/inference/engine.node.test.ts'`
Expected: FAIL — cannot find module `./engine`.

- [ ] **Step 3: Implement the engine**

`web/src/inference/engine.ts`:

```ts
import type { Chess } from "chess.js";
import { boardToTensor, indexToSquare, squareToIndex } from "./boardTensor";
import { eloToBucket } from "./elo";
import { legalFromMask, legalToMask } from "./legal";
import { maskedSoftmax, pickIndex } from "./sample";

type Session = { run(feeds: Record<string, any>): Promise<Record<string, { data: Float32Array }>> };
type OrtLike = {
  InferenceSession: { create(p: string | ArrayBuffer): Promise<Session> };
  Tensor: new (type: string, data: ArrayLike<number> | BigInt64Array, dims: number[]) => any;
};

export class Engine {
  private constructor(
    private ort: OrtLike, private enc: Session, private fh: Session, private th: Session,
    private nEloBuckets: number,
  ) {}

  static async load(ort: OrtLike, urls: { encode: string; fromHead: string; toHead: string },
                    meta: { nEloBuckets: number }): Promise<Engine> {
    const [enc, fh, th] = await Promise.all([
      ort.InferenceSession.create(urls.encode),
      ort.InferenceSession.create(urls.fromHead),
      ort.InferenceSession.create(urls.toHead),
    ]);
    return new Engine(ort, enc, fh, th, meta.nEloBuckets);
  }

  private elo(elo: number) {
    return new this.ort.Tensor("int64", BigInt64Array.from([BigInt(eloToBucket(elo, this.nEloBuckets))]), [1]);
  }

  private async squares(board: Chess) {
    const bt = boardToTensor(board);
    const out = await this.enc.run({ board_tensor: new this.ort.Tensor("float32", bt, [1, 8, 8, 18]) });
    return out["squares"];
  }

  async policy(board: Chess, elo: number) {
    const sq = await this.squares(board);
    const eloT = this.elo(elo);
    const fl = (await this.fh.run({ squares: sq, elo_idx: eloT }))["from_logits"].data;
    const fromProbs = maskedSoftmax(fl, legalFromMask(board), 1.0);
    const fromSq = pickIndex(fromProbs, { greedy: true });
    const fsqT = new this.ort.Tensor("int64", BigInt64Array.from([BigInt(fromSq)]), [1]);
    const tl = (await this.th.run({ squares: sq, from_sq: fsqT, elo_idx: eloT }))["to_logits"].data;
    const toProbs = maskedSoftmax(tl, legalToMask(board, fromSq), 1.0);
    const toSq = pickIndex(toProbs, { greedy: true });
    return { fromProbs, fromSq, toProbs, toSq };
  }

  async chooseMove(board: Chess, elo: number, opts: { temperature: number; greedy?: boolean; rand?: () => number }) {
    const sq = await this.squares(board);
    const eloT = this.elo(elo);
    const fl = (await this.fh.run({ squares: sq, elo_idx: eloT }))["from_logits"].data;
    const fromSq = pickIndex(maskedSoftmax(fl, legalFromMask(board), opts.temperature),
                             { greedy: !!opts.greedy, rand: opts.rand });
    const fsqT = new this.ort.Tensor("int64", BigInt64Array.from([BigInt(fromSq)]), [1]);
    const tl = (await this.th.run({ squares: sq, from_sq: fsqT, elo_idx: eloT }))["to_logits"].data;
    const toSq = pickIndex(maskedSoftmax(tl, legalToMask(board, fromSq), opts.temperature),
                           { greedy: !!opts.greedy, rand: opts.rand });
    const from = indexToSquare(fromSq), to = indexToSquare(toSq);
    // promotion: if a pawn reaches the last rank, default to queen (matches play.py)
    const needsPromo = board.moves({ verbose: true })
      .some((m) => m.from === from && m.to === to && m.promotion);
    return needsPromo ? { from, to, promotion: "q" as const } : { from, to };
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec <container> bash -lc 'cd /workspaces/eloquent-encoding/web && npx vitest run src/inference/engine.node.test.ts'`
Expected: PASS. (Requires `web/public/encode_int8.onnx`, `from_head_int8.onnx`, `to_head_int8.onnx` present from Task 3.)

- [ ] **Step 5: Commit**

```bash
git add web/src/inference/engine.ts web/src/inference/engine.node.test.ts
git commit -m "feat: Engine two-stage inference with int8 end-to-end parity"
```

---

### Task 11: Playable board (human vs bot)

**Files:**
- Create: `web/src/components/BoardPanel.tsx`
- Modify: `web/src/App.tsx`
- Create: `web/src/useEngine.ts` (loads the Engine with `onnxruntime-web` from `public/`)
- Test: `web/src/inference/engine.game.test.ts` (node: a full self-play game terminates with legal moves)

**Interfaces:**
- Consumes: `Engine`.
- Produces: `useEngine(): { engine: Engine | null }` (React hook, loads via `onnxruntime-web`); `BoardPanel` rendering `react-chessboard`, handling human drag-drop, then calling `engine.chooseMove` for the bot.

- [ ] **Step 1: Write the failing test (engine drives a legal, terminating game)**

`web/src/inference/engine.game.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import * as ort from "onnxruntime-node";
import { Chess } from "chess.js";
import { Engine } from "./engine";
import { mulberry32 } from "./rng";
import fixtures from "./__fixtures__/cases.json";

describe("engine self-play", () => {
  it("plays only legal moves and terminates", async () => {
    const eng = await Engine.load(ort as any, {
      encode: "public/encode_int8.onnx", fromHead: "public/from_head_int8.onnx", toHead: "public/to_head_int8.onnx",
    }, { nEloBuckets: fixtures.n_elo_buckets });
    const board = new Chess();
    const rand = mulberry32(7);
    let plies = 0;
    while (!board.isGameOver() && plies < 400) {
      const mv = await eng.chooseMove(board, 1500, { temperature: 1.0, greedy: false, rand });
      const res = board.move(mv);
      expect(res).not.toBeNull();   // legal
      plies++;
    }
    expect(plies).toBeGreaterThan(2);
  }, 60000);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec <container> bash -lc 'cd /workspaces/eloquent-encoding/web && npx vitest run src/inference/engine.game.test.ts'`
Expected: FAIL (or error) until the engine handles the full loop. If `board.move(mv)` ever returns null, the promotion/legality path needs the fix already in `chooseMove`; debug with systematic-debugging.

- [ ] **Step 3: Implement the UI + hook**

`web/src/useEngine.ts`:

```tsx
import { useEffect, useState } from "react";
import * as ort from "onnxruntime-web";
import { Engine } from "./inference/engine";

export function useEngine() {
  const [engine, setEngine] = useState<Engine | null>(null);
  useEffect(() => {
    const base = import.meta.env.BASE_URL;
    fetch(base + "model_meta.json").then((r) => r.json()).then((meta) =>
      Engine.load(ort as any, {
        encode: base + "encode_int8.onnx",
        fromHead: base + "from_head_int8.onnx",
        toHead: base + "to_head_int8.onnx",
      }, { nEloBuckets: meta.n_elo_buckets }).then(setEngine));
  }, []);
  return { engine };
}
```

`web/src/components/BoardPanel.tsx`:

```tsx
import React, { useCallback, useState } from "react";
import { Chessboard } from "react-chessboard";
import { Chess } from "chess.js";
import type { Engine } from "../inference/engine";

export function BoardPanel({ engine, elo, temperature }:
  { engine: Engine | null; elo: number; temperature: number }) {
  const [game, setGame] = useState(new Chess());
  const [thinking, setThinking] = useState(false);

  const botMove = useCallback(async (g: Chess) => {
    if (!engine || g.isGameOver()) return;
    setThinking(true);
    const mv = await engine.chooseMove(g, elo, { temperature, greedy: false });
    g.move(mv);
    setGame(new Chess(g.fen()));
    setThinking(false);
  }, [engine, elo, temperature]);

  const onDrop = useCallback((from: string, to: string) => {
    const g = new Chess(game.fen());
    const res = g.move({ from, to, promotion: "q" });
    if (!res) return false;
    setGame(g);
    void botMove(new Chess(g.fen()));
    return true;
  }, [game, botMove]);

  return (
    <div>
      <Chessboard position={game.fen()} onPieceDrop={onDrop} arePiecesDraggable={!thinking} />
      <button onClick={() => setGame(new Chess())}>New game</button>
      {game.isGameOver() && <p>Game over: {game.isCheckmate() ? "checkmate" : "draw"}</p>}
    </div>
  );
}
```

`web/src/App.tsx`:

```tsx
import React, { useState } from "react";
import { useEngine } from "./useEngine";
import { BoardPanel } from "./components/BoardPanel";

export function App() {
  const { engine } = useEngine();
  const [elo] = useState(1500);
  const [temperature] = useState(1.0);
  return (
    <div style={{ maxWidth: 560, margin: "0 auto" }}>
      <h1>Eloquent Bot</h1>
      {!engine && <p>Loading model…</p>}
      <BoardPanel engine={engine} elo={elo} temperature={temperature} />
    </div>
  );
}
```

- [ ] **Step 4: Run tests + build**

Run: `docker exec <container> bash -lc 'cd /workspaces/eloquent-encoding/web && npx vitest run src/inference/engine.game.test.ts && npm run build'`
Expected: self-play test PASS; production build succeeds.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/BoardPanel.tsx web/src/App.tsx web/src/useEngine.ts web/src/inference/engine.game.test.ts
git commit -m "feat: playable human-vs-bot board"
```

---

### Task 12: Elo + temperature controls

**Files:**
- Create: `web/src/components/Controls.tsx`
- Modify: `web/src/App.tsx`

**Interfaces:**
- Produces: `Controls({ elo, setElo, temperature, setTemperature })` — an elo slider (600–2400, step 100) and a temperature slider (0.1–2.0, step 0.1), with the temperature endpoints labelled "sharp / human / wild".

- [ ] **Step 1: Implement controls**

`web/src/components/Controls.tsx`:

```tsx
import React from "react";

export function Controls({ elo, setElo, temperature, setTemperature }: {
  elo: number; setElo: (n: number) => void; temperature: number; setTemperature: (n: number) => void;
}) {
  return (
    <div style={{ display: "flex", gap: 24, margin: "12px 0" }}>
      <label>Elo: {elo}
        <input type="range" min={600} max={2400} step={100} value={elo}
               onChange={(e) => setElo(Number(e.target.value))} />
      </label>
      <label>Temperature: {temperature.toFixed(1)}
        <input type="range" min={0.1} max={2.0} step={0.1} value={temperature}
               onChange={(e) => setTemperature(Number(e.target.value))} />
      </label>
    </div>
  );
}
```

`web/src/App.tsx` — wire state and the controls:

```tsx
import React, { useState } from "react";
import { useEngine } from "./useEngine";
import { BoardPanel } from "./components/BoardPanel";
import { Controls } from "./components/Controls";

export function App() {
  const { engine } = useEngine();
  const [elo, setElo] = useState(1500);
  const [temperature, setTemperature] = useState(1.0);
  return (
    <div style={{ maxWidth: 560, margin: "0 auto" }}>
      <h1>Eloquent Bot</h1>
      {!engine && <p>Loading model…</p>}
      <Controls elo={elo} setElo={setElo} temperature={temperature} setTemperature={setTemperature} />
      <BoardPanel engine={engine} elo={elo} temperature={temperature} />
    </div>
  );
}
```

- [ ] **Step 2: Build to verify**

Run: `docker exec <container> bash -lc 'cd /workspaces/eloquent-encoding/web && npm run build'`
Expected: build succeeds; changing the elo slider changes the bot's elo bucket on its next move (verified visually in `npm run dev`).

- [ ] **Step 3: Commit**

```bash
git add web/src/components/Controls.tsx web/src/App.tsx
git commit -m "feat: elo + temperature controls"
```

---

### Task 13: "What the model is thinking" panel

**Files:**
- Create: `web/src/components/ThinkingPanel.tsx`
- Create: `web/src/inference/topMoves.ts`
- Modify: `web/src/components/BoardPanel.tsx` (lift the live `Chess` position up; pass to the panel; add square highlight styles), `web/src/App.tsx`
- Test: `web/src/inference/topMoves.test.ts`

**Interfaces:**
- Consumes: `Engine`, `maskedSoftmax`, `legalFromMask`, `legalToMask`, `indexToSquare`.
- Produces: `topMoves(engine, board, elo, k): Promise<{ uci: string; san: string; prob: number }[]>` — full joint `P(from)·P(to|from)` over legal moves, top-k. (Mirrors the Python top-k done earlier in the session.)
- Produces: `ThinkingPanel({ moves })` rendering probability bars; and from-square highlight styles for the board heatmap.

- [ ] **Step 1: Write the failing test**

`web/src/inference/topMoves.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import * as ort from "onnxruntime-node";
import { Chess } from "chess.js";
import { Engine } from "./engine";
import { topMoves } from "./topMoves";
import fixtures from "./__fixtures__/cases.json";

describe("topMoves", () => {
  it("returns sorted legal moves with probabilities summing <= 1", async () => {
    const eng = await Engine.load(ort as any, {
      encode: "public/encode_int8.onnx", fromHead: "public/from_head_int8.onnx", toHead: "public/to_head_int8.onnx",
    }, { nEloBuckets: fixtures.n_elo_buckets });
    const moves = await topMoves(eng, new Chess(), 1500, 5);
    expect(moves.length).toBe(5);
    for (let i = 1; i < moves.length; i++) expect(moves[i - 1].prob).toBeGreaterThanOrEqual(moves[i].prob);
    expect(moves.every((m) => new Chess().move(m.san) !== null)).toBe(true);
  }, 60000);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec <container> bash -lc 'cd /workspaces/eloquent-encoding/web && npx vitest run src/inference/topMoves.test.ts'`
Expected: FAIL — cannot find module `./topMoves`.

- [ ] **Step 3: Implement**

`web/src/inference/topMoves.ts`:

```ts
import type { Chess } from "chess.js";
import type { Engine } from "./engine";
import { maskedSoftmax } from "./sample";
import { legalFromMask, legalToMask } from "./legal";
import { indexToSquare, squareToIndex } from "./boardTensor";

// Needs raw logits per from-square; expose a helper on Engine via policy-style calls.
export async function topMoves(engine: Engine, board: Chess, elo: number, k: number) {
  const dist = await engine.distributions(board, elo);   // see Engine.distributions below
  const fromMask = legalFromMask(board);
  const fromProbs = maskedSoftmax(dist.fromLogits, fromMask, 1.0);
  const out: { uci: string; san: string; prob: number }[] = [];
  for (let f = 0; f < 64; f++) {
    if (!fromMask[f]) continue;
    const toLogits = await dist.toLogits(f);
    const toProbs = maskedSoftmax(toLogits, legalToMask(board, f), 1.0);
    for (let t = 0; t < 64; t++) {
      if (toProbs[t] <= 0) continue;
      const uci = indexToSquare(f) + indexToSquare(t);
      const probe = new (board.constructor as any)(board.fen());
      const mv = probe.move({ from: indexToSquare(f), to: indexToSquare(t), promotion: "q" });
      if (!mv) continue;
      out.push({ uci, san: mv.san, prob: fromProbs[f] * toProbs[t] });
    }
  }
  out.sort((a, b) => b.prob - a.prob);
  return out.slice(0, k);
}
```

Add to `web/src/inference/engine.ts` (a small accessor that returns logits and a lazy to-head closure, reusing the cached squares):

```ts
  async distributions(board: Chess, elo: number) {
    const sq = await this.squares(board);
    const eloT = this.elo(elo);
    const fromLogits = (await this.fh.run({ squares: sq, elo_idx: eloT }))["from_logits"].data;
    const toLogits = async (fromSq: number) => {
      const fsqT = new this.ort.Tensor("int64", BigInt64Array.from([BigInt(fromSq)]), [1]);
      return (await this.th.run({ squares: sq, from_sq: fsqT, elo_idx: eloT }))["to_logits"].data;
    };
    return { fromLogits, toLogits };
  }
```

`web/src/components/ThinkingPanel.tsx`:

```tsx
import React from "react";

export function ThinkingPanel({ moves }: { moves: { san: string; prob: number }[] }) {
  const max = moves.length ? moves[0].prob : 1;
  return (
    <div style={{ minWidth: 180 }}>
      <h3>Model's top moves</h3>
      {moves.map((m) => (
        <div key={m.san} style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ width: 48 }}>{m.san}</span>
          <div style={{ background: "#4a90d9", height: 12, width: `${(m.prob / max) * 100}%` }} />
          <span>{(m.prob * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}
```

In `BoardPanel.tsx`, after each position settles, compute `topMoves(engine, game, elo, 5)`, store in state, render `<ThinkingPanel>` beside the board, and set `customSquareStyles` to tint the top move's from/to squares (e.g. `{ [from]: { background: "rgba(74,144,217,0.5)" }}`). Lift `game`/`elo`/`temperature` so the panel updates live.

- [ ] **Step 4: Run tests + build**

Run: `docker exec <container> bash -lc 'cd /workspaces/eloquent-encoding/web && npx vitest run src/inference/topMoves.test.ts && npm run build'`
Expected: PASS; build succeeds.

- [ ] **Step 5: Commit**

```bash
git add web/src/inference/topMoves.ts web/src/inference/topMoves.test.ts web/src/inference/engine.ts web/src/components/ThinkingPanel.tsx web/src/components/BoardPanel.tsx web/src/App.tsx
git commit -m "feat: model-thinking panel (top-move bars + board heatmap)"
```

---

### Task 14: GitHub Pages deploy + committed model + storage docs

**Files:**
- Create: `.github/workflows/pages.yml`
- Modify: `web/.gitignore` (un-ignore `public/*.onnx`), `web/vite.config.ts` (Pages base path)
- Create: `web/README.md` (storage policy + how to re-export)
- Commit: `web/public/encode_int8.onnx`, `web/public/from_head_int8.onnx`, `web/public/to_head_int8.onnx`, `web/public/model_meta.json`

**Interfaces:** none (deploy + assets).

- [ ] **Step 1: Set the Pages base path**

`web/vite.config.ts` — set the repo base so assets resolve under `https://<user>.github.io/<repo>/`:

```ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
// On Pages the site is served from /<repo>/; locally base stays "/".
export default defineConfig({ plugins: [react()], base: process.env.VITE_BASE ?? "/" });
```

(The workflow sets `VITE_BASE=/eloquent-encoding/`.)

- [ ] **Step 2: Commit the model artifacts**

Confirm `web/public/*_int8.onnx` exist (from Task 3) and total < 10MB. Ensure `web/.gitignore` does not exclude them (it only ignores `node_modules/` and `dist/`).

```bash
docker exec <container> bash -lc 'cd /workspaces/eloquent-encoding && ls -lh web/public/*.onnx && du -ch web/public/*.onnx | tail -1'
```

- [ ] **Step 3: Add the deploy workflow**

`.github/workflows/pages.yml`:

```yaml
name: Deploy web bot to Pages
on:
  push:
    branches: [main]
    paths: ["web/**", ".github/workflows/pages.yml"]
  workflow_dispatch:
permissions:
  contents: read
  pages: write
  id-token: write
concurrency:
  group: pages
  cancel-in-progress: true
jobs:
  build:
    runs-on: ubuntu-latest
    defaults: { run: { working-directory: web } }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: "20", cache: "npm", cache-dependency-path: web/package-lock.json }
      - run: npm ci
      - run: npm test
      - run: npm run build
        env: { VITE_BASE: "/eloquent-encoding/" }
      - uses: actions/upload-pages-artifact@v3
        with: { path: web/dist }
  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment: { name: github-pages, url: "${{ steps.deployment.outputs.page_url }}" }
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 4: Document the storage policy**

`web/README.md`:

```markdown
# Eloquent web bot

Client-side chess bot (runs the policy model in-browser via onnxruntime-web).

## Model weights
- Deployed model: int8 ONNX in `public/*_int8.onnx` (~7MB total), committed directly.
- Re-export after retraining: `python scripts/export_onnx.py --checkpoint <ckpt>` then
  `python scripts/gen_web_fixtures.py` and run `npm test` to re-verify parity.
- **Do not** commit `.pt` checkpoints or use Git LFS (Pages serves the LFS pointer, not the
  binary). If the deployed model starts changing frequently, move weights to GitHub Release
  assets and fetch them by URL in `useEngine.ts` instead of committing.

## Dev
`npm install && npm run dev`. Tests: `npm test`.
```

- [ ] **Step 5: Verify build, commit, and enable Pages**

Run: `docker exec <container> bash -lc 'cd /workspaces/eloquent-encoding/web && VITE_BASE=/eloquent-encoding/ npm run build && ls dist'`
Expected: `dist/` contains `index.html`, hashed JS, and the `.onnx` files under `dist/`.

```bash
git add .github/workflows/pages.yml web/vite.config.ts web/README.md web/public/encode_int8.onnx web/public/from_head_int8.onnx web/public/to_head_int8.onnx web/public/model_meta.json
git commit -m "feat: GitHub Pages deploy workflow + committed int8 model"
```

After pushing, enable Pages in the repo settings (Source: GitHub Actions). The first `push` to `main` touching `web/**` deploys the site.

---

## Self-Review

**Spec coverage:**
- Client-side, no backend → Tasks 1–3 (ONNX), 5–10 (in-browser engine), 14 (static Pages). ✓
- Elo selection → Task 12. ✓
- "What the model is thinking" plots → Task 13 (top-move bars + board heatmap). ✓
- Weight storage watched → Task 3 (size gate), Task 14 (commit policy, LFS trap, Releases escape hatch, README). ✓
- Ships `base_64M_stage_1` → Global Constraints + Tasks 2/4. ✓
- Parity (the risk) → fp32 (Task 2), int8 top-1 (Task 3), cross-language end-to-end (Task 10). ✓

**Placeholder scan:** No TODO/TBD; every code step has full code; every test has assertions; commands have expected output.

**Type consistency:** `boardToTensor`/`squareToIndex`/`indexToSquare` (Task 6) reused in 8/10/13; `eloToBucket(elo,n)` (Task 7) used in `Engine.elo` (10); `maskedSoftmax`/`pickIndex` signatures (9) used in 10/13; `Engine.load/policy/chooseMove/distributions` consistent across 10/11/13; ONNX I/O names (`board_tensor`,`squares`,`elo_idx`,`from_sq`,`from_logits`,`to_logits`) consistent between `export_onnx.py` (2) and `engine.ts` (10).

**Known risk flagged for the implementer:** ONNX export of `nn.TransformerEncoder` — if the nested-tensor disable in Task 1 is insufficient, the fallback is `torch.onnx.export(..., dynamo=False)` (already noted) or tracing each `TransformerEncoderLayer`. The Task 2 parity test is the gate that catches any silent numeric divergence.
