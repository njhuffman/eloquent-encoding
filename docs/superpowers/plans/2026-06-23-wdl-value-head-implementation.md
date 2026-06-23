# WDL Value Head — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a jointly-trained, human-realized WDL value head (P(win/draw/loss | board, mover_elo)) to the two-stage policy, validate at 16M, then scale to 64M.

**Architecture:** Regenerate the packed dataset with two new per-row columns (`result`, `opp_elo`); add a `WDLHead` MLP on the encoder's (currently unused) CLS token, conditioned on the mover's elo; train jointly with loss `from_ce + to_ce + λ·wdl_ce`.

**Tech Stack:** Python, PyTorch, h5py, python-chess; existing `style_policy` + `dataset_generation` packages.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-06-23-wdl-value-head-design.md`.
- `result` (int8) is the outcome **from the side-to-move's perspective**: `loss=0, draw=1, win=2`. WDL head logits are in this same order.
- `opp_elo` (int16) is stored but **not consumed by v1** (the value head conditions on the mover's elo only).
- Value head input = encoder **CLS token** (`encode()` returns `(cls, squares)`; CLS is currently discarded everywhere — the value head is its first consumer). Mover-elo embedding is value-head-specific (mirrors `FromHead`/`ToHead`).
- Joint loss `from_ce + to_ce + λ·wdl_ce`, `λ` = config `value_loss_weight`, default **1.0**.
- Packed training schema columns: `packed_pre (uint8,34)`, `from_legal_u64 (uint64)`, `to_legal_u64 (uint64)`, `from_sq/to_sq/promotion (uint8)`, `elo_to_move (int16)`, plus new `opp_elo (int16)`, `result (int8)`. **Drop `packed_post`** (training never reads it).
- Regen writes NEW files (`wdl_*.h5`) — never overwrite the existing `j3_*.h5`.
- Source data + outputs live at `/mnt/eloquence_bulk/databases/`. Tests/python run in the container: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 pytest ...'` (pytest = `pytest` or `/usr/local/py-utils/bin/pytest`, NOT `python -m pytest`).
- Reuse current codecs: `style_policy.board_encode.board_to_packed/legal_from_u64/legal_to_u64`, `move_predictor.encoding.move_to_from_to/promotion_code`. Do not revive archived jepa3 build code.
- 16M go/no-go gate: WDL val log-loss beats an elo-only prior AND policy full-move top-1 is neutral-or-better vs. `base_16M` re-evaluated on the same val set.

---

## File Structure

- `dataset_generation/candidate_collect.py` (modify) — emit `opp_elo` + `result` per row; drop unterminated games.
- `dataset_generation/hdf5_io.py` (modify) — add `PackedBatchWriter` (packed schema + new columns).
- `dataset_generation/builder.py` (modify) — compute packed columns + result/opp_elo per sampled row; use `PackedBatchWriter`; fix row-count check.
- `dataset_generation/wdl_training_16M.yaml`, `wdl_validation_1M.yaml`, `wdl_training_64M.yaml` (create) — recipes.
- `style_policy/dataset.py` (modify) — return `result` (and `opp_elo`) in the batch dict.
- `style_policy/value_head.py` (create) — `WDLHead`.
- `style_policy/model.py` (modify) — build `WDLHead` in `from_config`; `forward_value`; value logits from `forward_policy`.
- `style_policy/loss.py` (modify) — `wdl_ce` + `wdl_accuracy`.
- `style_policy/training_loop.py` (modify) — joint loss with `value_loss_weight`; val WDL metrics; W&B value logging.
- `style_policy/model_configs/wdl_16M.yaml`, `wdl_64M.yaml` (create) — training configs.
- `scripts/eval_wdl.py` (create) — WDL calibration/log-loss vs elo-prior + policy full_top1.
- Tests under `tests/dataset_generation/` and `tests/style_policy/`.

---

### Task 1: Emit `opp_elo` + `result` from candidate collection

**Files:**
- Modify: `dataset_generation/candidate_collect.py`
- Test: `tests/dataset_generation/test_candidate_collect_result.py`

**Interfaces:**
- Produces: `collect_candidate_positions(game, *, skip_opening_plies, exclude_single_legal_move) -> tuple[list[chess.Move], list[tuple[int,int,int,int,int,chess.Move]]]` where each row is `(ply, side_to_move, elo_to_move, opp_elo, result, played_move)`. `result` is `loss=0/draw=1/win=2` from the side-to-move's perspective. Games with `Result` not in `{1-0,0-1,1/2-1/2}` return `([], [])`.

- [ ] **Step 1: Write the failing test**

`tests/dataset_generation/test_candidate_collect_result.py`:

```python
import io
import chess.pgn
from dataset_generation.candidate_collect import collect_candidate_positions

_PGN = """[White "a"]
[Black "b"]
[WhiteElo "1500"]
[BlackElo "1600"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0
"""

def _game():
    return chess.pgn.read_game(io.StringIO(_PGN))

def test_rows_carry_opp_elo_and_result_from_stm_perspective():
    _, rows = collect_candidate_positions(_game(), skip_opening_plies=0, exclude_single_legal_move=False)
    assert rows, "expected candidate rows"
    # White (stm=0) won -> result 2, opp_elo=1600; Black (stm=1) lost -> result 0, opp_elo=1500
    for ply, stm, elo, opp, result, move in rows:
        if stm == 0:
            assert elo == 1500 and opp == 1600 and result == 2
        else:
            assert elo == 1600 and opp == 1500 and result == 0

def test_unterminated_game_dropped():
    pgn = _PGN.replace('[Result "1-0"]', '[Result "*"]').replace(" 1-0", " *")
    _, rows = collect_candidate_positions(chess.pgn.read_game(io.StringIO(pgn)),
                                          skip_opening_plies=0, exclude_single_legal_move=False)
    assert rows == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 pytest tests/dataset_generation/test_candidate_collect_result.py -q'`
Expected: FAIL — rows are 4-tuples / ValueError on unpack, or `tests/dataset_generation` import error (create `tests/dataset_generation/__init__.py` if needed — match the existing `tests/style_policy/` layout, which has no `__init__.py`, so none is needed).

- [ ] **Step 3: Implement**

Replace the body of `collect_candidate_positions` in `dataset_generation/candidate_collect.py` (keep `_parse_elo`, `board_at_ply`):

```python
_WHITE_WDL = {"1-0": 2, "1/2-1/2": 1, "0-1": 0}  # White's win/draw/loss


def collect_candidate_positions(
    game: chess.pgn.Game,
    *,
    skip_opening_plies: int,
    exclude_single_legal_move: bool,
) -> tuple[list[chess.Move], list[tuple[int, int, int, int, int, chess.Move]]]:
    """Mainline-order candidates before each half-move.

    Returns ``mainline`` and rows ``(ply, side_to_move 0/1, elo_to_move, opp_elo, result, played_move)``.
    ``result`` is loss=0/draw=1/win=2 from the side-to-move's perspective. Games whose
    Result header is not a terminated outcome (e.g. ``*``) are dropped (returns ``[], []``).
    """
    white_elo_h = _parse_elo(game.headers, "white")
    black_elo_h = _parse_elo(game.headers, "black")
    if white_elo_h is None or black_elo_h is None:
        return [], []
    white_wdl = _WHITE_WDL.get(game.headers.get("Result", "*"))
    if white_wdl is None:
        return [], []

    mainline = list(game.mainline_moves())
    board = game.board()
    ply = 0
    out: list[tuple[int, int, int, int, int, chess.Move]] = []
    for move in mainline:
        if ply >= skip_opening_plies:
            if exclude_single_legal_move and board.legal_moves.count() < 2:
                board.push(move)
                ply += 1
                continue
            if board.turn == chess.WHITE:
                stm, elo_tm, opp_elo, result = 0, white_elo_h, black_elo_h, white_wdl
            else:
                stm, elo_tm, opp_elo, result = 1, black_elo_h, white_elo_h, 2 - white_wdl
            out.append((ply, stm, elo_tm, opp_elo, result, move))
        board.push(move)
        ply += 1
    return mainline, out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 pytest tests/dataset_generation/test_candidate_collect_result.py -q'`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add dataset_generation/candidate_collect.py tests/dataset_generation/test_candidate_collect_result.py
git commit -m "feat: emit opp_elo + result (STM perspective) from candidate collection"
```

---

### Task 2: `PackedBatchWriter` (packed schema + result/opp_elo)

**Files:**
- Modify: `dataset_generation/hdf5_io.py`
- Test: `tests/dataset_generation/test_packed_writer.py`

**Interfaces:**
- Produces: `PackedBatchWriter(path, batch_size=1024)` context manager with
  `append_row(*, packed_pre: np.ndarray(34,uint8), from_legal_u64: int, to_legal_u64: int, from_sq: int, to_sq: int, promotion: int, elo_to_move: int, opp_elo: int, result: int)`. Writes datasets: `packed_pre (N,34) uint8`, `from_legal_u64/to_legal_u64 uint64`, `from_sq/to_sq/promotion uint8`, `elo_to_move/opp_elo int16`, `result int8`. Matches the schema `style_policy/dataset.py` reads, plus `opp_elo`/`result`.

- [ ] **Step 1: Write the failing test**

`tests/dataset_generation/test_packed_writer.py`:

```python
import numpy as np
import h5py
from dataset_generation.hdf5_io import PackedBatchWriter

def test_packed_writer_roundtrip(tmp_path):
    p = tmp_path / "x.h5"
    pre = np.arange(34, dtype=np.uint8)
    with PackedBatchWriter(p, batch_size=2) as w:
        for i in range(3):
            w.append_row(packed_pre=pre + i, from_legal_u64=(1 << 63) | 1, to_legal_u64=5,
                         from_sq=12, to_sq=28, promotion=0, elo_to_move=1500, opp_elo=1600, result=2)
    with h5py.File(p, "r") as f:
        assert f["packed_pre"].shape == (3, 34) and f["packed_pre"].dtype == np.uint8
        assert int(f["result"][0]) == 2 and str(f["result"].dtype) == "int8"
        assert int(f["opp_elo"][1]) == 1600
        assert np.uint64(f["from_legal_u64"][2]) == np.uint64((1 << 63) | 1)  # bit 63 preserved
        assert list(f["packed_pre"][1]) == list((pre + 1))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 pytest tests/dataset_generation/test_packed_writer.py -q'`
Expected: FAIL — `cannot import name 'PackedBatchWriter'`.

- [ ] **Step 3: Implement**

Add to `dataset_generation/hdf5_io.py` (alongside `SampleBatchWriter`; keep `CHUNK`):

```python
PACKED_LEN = 34


class PackedBatchWriter:
    """Append rows to a new HDF5 in the packed training schema (+ opp_elo, result)."""

    _SCALAR = (
        ("from_legal_u64", np.uint64), ("to_legal_u64", np.uint64),
        ("from_sq", np.uint8), ("to_sq", np.uint8), ("promotion", np.uint8),
        ("elo_to_move", np.int16), ("opp_elo", np.int16), ("result", np.int8),
    )
    COLUMNS = ("packed_pre",) + tuple(n for n, _ in _SCALAR)

    def __init__(self, path: Path, batch_size: int = 1024) -> None:
        self.path = path
        self.batch_size = batch_size
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = h5py.File(path, "w")
        self._n = 0
        self._f.create_dataset("packed_pre", shape=(0, PACKED_LEN), maxshape=(None, PACKED_LEN),
                               dtype=np.uint8, chunks=(CHUNK, PACKED_LEN), compression="gzip", compression_opts=4)
        for name, dt in self._SCALAR:
            self._f.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dt,
                                   chunks=(CHUNK,), compression="gzip", compression_opts=4)
        self._buf = {c: [] for c in self.COLUMNS}

    def __enter__(self) -> "PackedBatchWriter":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self.flush()
        self._f.attrs["row_count"] = self._n
        self._f.close()

    def flush(self) -> None:
        if not self._buf["packed_pre"]:
            return
        m = len(self._buf["packed_pre"])
        o = self._n
        for name in self.COLUMNS:
            d = self._f[name]
            if name == "packed_pre":
                d.resize((o + m, PACKED_LEN))
                d[o : o + m] = np.asarray(self._buf[name], dtype=np.uint8)
            else:
                d.resize((o + m,))
                d[o : o + m] = np.asarray(self._buf[name], dtype=d.dtype)
            self._buf[name].clear()
        self._n += m

    def append_row(self, *, packed_pre, from_legal_u64, to_legal_u64, from_sq, to_sq,
                   promotion, elo_to_move, opp_elo, result) -> None:
        self._buf["packed_pre"].append(np.asarray(packed_pre, dtype=np.uint8).reshape(PACKED_LEN))
        self._buf["from_legal_u64"].append(np.uint64(from_legal_u64))
        self._buf["to_legal_u64"].append(np.uint64(to_legal_u64))
        self._buf["from_sq"].append(from_sq)
        self._buf["to_sq"].append(to_sq)
        self._buf["promotion"].append(promotion)
        self._buf["elo_to_move"].append(elo_to_move)
        self._buf["opp_elo"].append(opp_elo)
        self._buf["result"].append(result)
        if len(self._buf["packed_pre"]) >= self.batch_size:
            self.flush()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 pytest tests/dataset_generation/test_packed_writer.py -q'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dataset_generation/hdf5_io.py tests/dataset_generation/test_packed_writer.py
git commit -m "feat: PackedBatchWriter (packed training schema + opp_elo/result)"
```

---

### Task 3: Wire builder to write packed rows + result/opp_elo

**Files:**
- Modify: `dataset_generation/builder.py`
- Test: `tests/dataset_generation/test_build_packed_e2e.py`

**Interfaces:**
- Consumes: `collect_candidate_positions` (6-tuple rows, Task 1), `PackedBatchWriter` (Task 2), `board_to_packed`/`legal_from_u64`/`legal_to_u64` (`style_policy.board_encode`), `move_to_from_to`/`promotion_code` (`move_predictor.encoding`).
- Produces: `build_from_recipe(recipe, *, data_dir, output_dir) -> Path` writing a packed h5 (the schema above). Row-count check reads `packed_pre`.

- [ ] **Step 1: Write the failing test (tiny end-to-end build)**

`tests/dataset_generation/test_build_packed_e2e.py`:

```python
import io
import h5py
import numpy as np
import zstandard
from pathlib import Path
from dataset_generation.recipe import Recipe
from dataset_generation.builder import build_from_recipe
from style_policy.packed_codec import packed_to_board_tensor

_GAME = """[Event "x"]
[White "a"]
[Black "b"]
[WhiteElo "1550"]
[BlackElo "1550"]
[TimeControl "600+0"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0

"""

_RECIPE = """name: wdl_smoke
master_seed: 1
time_control: 600+0
bucket_by: white
skip_opening_plies: 2
exclude_single_legal_move: false
source_plans:
  - source: smoke.pgn.zst
    strata:
      - {elo_min: 1500, elo_max: 1599, take_games: 1, samples_per_game: 3, stratum_seed: 1}
"""

def test_packed_build_has_result_and_valid_boards(tmp_path):
    data_dir = tmp_path / "data"; data_dir.mkdir()
    # 4 identical games so the single stratum's take_games quota is reachable
    raw = (_GAME * 4).encode()
    with open(data_dir / "smoke.pgn.zst", "wb") as fh:
        fh.write(zstandard.ZstdCompressor().compress(raw))
    (tmp_path / "wdl_smoke.yaml").write_text(_RECIPE)
    recipe = Recipe.load(tmp_path / "wdl_smoke.yaml")
    out = build_from_recipe(recipe, data_dir=data_dir, output_dir=tmp_path)
    with h5py.File(out, "r") as f:
        assert f["packed_pre"].shape == (3, 34)
        assert str(f["result"].dtype) == "int8"
        assert set(np.unique(f["result"])).issubset({0, 1, 2})
        # White won (1-0); rows where stm=white should be result 2. Decode a board to confirm validity.
        bt = packed_to_board_tensor(f["packed_pre"][0:3])
        assert bt.shape == (3, 8, 8, 18)
```

(Note: `Recipe.load` and field names are assumed from the 32M recipe; if `Recipe`'s API differs, adapt the test's recipe construction to match `dataset_generation/recipe.py` — do not change the recipe schema.)

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 pytest tests/dataset_generation/test_build_packed_e2e.py -q'`
Expected: FAIL — builder still writes `fen` (no `packed_pre`/`result`); KeyError on `f["packed_pre"]`.

- [ ] **Step 3: Implement**

In `dataset_generation/builder.py`:

(a) imports — replace `SampleBatchWriter` import and add codecs:

```python
from dataset_generation.hdf5_io import PackedBatchWriter
from style_policy.board_encode import board_to_packed, legal_from_u64, legal_to_u64
```

(b) `_write_samples_for_stratum` — unpack the 6-tuple, compute packed columns, write via `PackedBatchWriter`:

```python
    rng = _rng_for_game(master_seed, source_plan_index, stratum, stratum_index, g)
    for j in _sample_indices(rng, k, stratum.samples_per_game):
        ply, stm, elo, opp_elo, result, move = candidates[j]
        board = board_at_ply(mainline, ply)
        fr, to = move_to_from_to(move)
        writer.append_row(
            packed_pre=board_to_packed(board),
            from_legal_u64=legal_from_u64(board),
            to_legal_u64=legal_to_u64(board, fr),
            from_sq=fr, to_sq=to, promotion=promotion_code(move),
            elo_to_move=int(elo), opp_elo=int(opp_elo), result=int(result),
        )
```

(c) the `writer` type annotation in `_write_samples_for_stratum` signature: change `writer: SampleBatchWriter` → `writer: PackedBatchWriter`.

(d) `build_from_recipe` — use `PackedBatchWriter` and check `packed_pre`:

```python
        with PackedBatchWriter(out_path) as writer:
            ...
    expected = recipe.target_sample_rows()
    with h5py.File(out_path, "r") as f:
        n = int(f["packed_pre"].shape[0])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 pytest tests/dataset_generation/test_build_packed_e2e.py tests/test_pgn_stream_prefilter.py -q'`
Expected: PASS (new e2e test + pre-existing stream test still green).

- [ ] **Step 5: Commit**

```bash
git add dataset_generation/builder.py tests/dataset_generation/test_build_packed_e2e.py
git commit -m "feat: build packed dataset with result/opp_elo via PackedBatchWriter"
```

---

### Task 4: `PackedMoveDataset` returns `result`

**Files:**
- Modify: `style_policy/dataset.py`
- Test: `tests/style_policy/test_dataset_result.py`

**Interfaces:**
- Produces: `PackedMoveDataset.__getitem__` includes `"result": torch.tensor(int, int64)` (and `"opp_elo"`), in addition to existing keys. `collate` unchanged (generic stack).

- [ ] **Step 1: Write the failing test**

`tests/style_policy/test_dataset_result.py`:

```python
import numpy as np, h5py, torch
from dataset_generation.hdf5_io import PackedBatchWriter
from style_policy.dataset import PackedMoveDataset

def _tiny(path):
    with PackedBatchWriter(path, batch_size=4) as w:
        for i in range(4):
            w.append_row(packed_pre=np.zeros(34, np.uint8), from_legal_u64=1, to_legal_u64=1,
                         from_sq=0, to_sq=1, promotion=0, elo_to_move=1500, opp_elo=1400, result=i % 3)

def test_dataset_exposes_result(tmp_path):
    p = tmp_path / "d.h5"; _tiny(p)
    ds = PackedMoveDataset(p)
    item = ds[2]
    assert "result" in item and int(item["result"]) == 2
    batch = PackedMoveDataset.collate([ds[0], ds[1], ds[2]])
    assert batch["result"].tolist() == [0, 1, 2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 pytest tests/style_policy/test_dataset_result.py -q'`
Expected: FAIL — `KeyError: 'result'`.

- [ ] **Step 3: Implement**

In `style_policy/dataset.py`, add `result` (and `opp_elo`) to the `__getitem__` dict (after the existing `elo_to_move` line):

```python
            "elo_to_move": torch.tensor(int(f["elo_to_move"][idx]), dtype=torch.int64),
            "result": torch.tensor(int(f["result"][idx]), dtype=torch.int64),
            "opp_elo": torch.tensor(int(f["opp_elo"][idx]), dtype=torch.int64),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 pytest tests/style_policy/test_dataset_result.py -q'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add style_policy/dataset.py tests/style_policy/test_dataset_result.py
git commit -m "feat: PackedMoveDataset exposes result + opp_elo"
```

---

### Task 5: `WDLHead`

**Files:**
- Create: `style_policy/value_head.py`
- Test: `tests/style_policy/test_value_head.py`

**Interfaces:**
- Produces: `WDLHead(*, d_model, hidden, elo_dim=0, n_elo_buckets=0)` with `forward(cls: (B,d_model), *, elo_idx: (B,)|None) -> (B,3)` raw WDL logits (order loss/draw/win). Own `elo_emb = Embedding(n_elo_buckets+1, elo_dim)`; `elo_idx=None` → null index `n_elo_buckets`.

- [ ] **Step 1: Write the failing test**

`tests/style_policy/test_value_head.py`:

```python
import torch
from style_policy.value_head import WDLHead

def test_shapes_and_elo_conditioning():
    h = WDLHead(d_model=32, hidden=16, elo_dim=8, n_elo_buckets=40).eval()
    cls = torch.randn(5, 32)
    out = h(cls, elo_idx=torch.tensor([0, 1, 2, 3, 4]))
    assert out.shape == (5, 3)
    # null-elo path works and differs from a real bucket
    out_null = h(cls, elo_idx=None)
    assert out_null.shape == (5, 3)
    assert not torch.allclose(out, out_null)

def test_no_elo_variant():
    h = WDLHead(d_model=32, hidden=16, elo_dim=0, n_elo_buckets=0).eval()
    assert h(torch.randn(2, 32), elo_idx=None).shape == (2, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 pytest tests/style_policy/test_value_head.py -q'`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

`style_policy/value_head.py`:

```python
"""WDL value head over the encoder CLS token (+ optional mover-elo conditioning).

Returns RAW 3-logit win/draw/loss scores in the order (loss=0, draw=1, win=2),
matching the `result` label encoding. Mirrors the elo-embedding pattern in
policy_heads.py (own embedding, extra index for unknown elo)."""
from __future__ import annotations
import torch
import torch.nn as nn


class WDLHead(nn.Module):
    def __init__(self, *, d_model: int, hidden: int, elo_dim: int = 0, n_elo_buckets: int = 0):
        super().__init__()
        self.elo_dim = int(elo_dim)
        self.null_elo = int(n_elo_buckets)
        if elo_dim > 0:
            self.elo_emb = nn.Embedding(n_elo_buckets + 1, elo_dim)
        self.score = nn.Sequential(
            nn.Linear(d_model + self.elo_dim, hidden), nn.GELU(), nn.Linear(hidden, 3))

    def forward(self, cls: torch.Tensor, *, elo_idx: torch.Tensor | None = None) -> torch.Tensor:
        b = cls.shape[0]
        if self.elo_dim > 0:
            if elo_idx is None:
                elo_idx = torch.full((b,), self.null_elo, device=cls.device, dtype=torch.long)
            cls = torch.cat([cls, self.elo_emb(elo_idx)], dim=-1)
        return self.score(cls)  # (B,3)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 pytest tests/style_policy/test_value_head.py -q'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add style_policy/value_head.py tests/style_policy/test_value_head.py
git commit -m "feat: WDLHead (CLS + mover-elo -> 3 WDL logits)"
```

---

### Task 6: Wire `WDLHead` into `BasePolicy`

**Files:**
- Modify: `style_policy/model.py`
- Test: `tests/style_policy/test_model_value.py`

**Interfaces:**
- Consumes: `WDLHead` (Task 5).
- Produces:
  - `BasePolicy.__init__` gains a `value_head` arg; `from_config` builds `WDLHead(d_model=d, hidden=h, elo_dim=elo_dim, n_elo_buckets=n_elo)`.
  - `forward_value(packed_pre, *, elo_idx=None) -> (B,3)` WDL logits from the CLS token.
  - `forward_policy(packed_pre, from_sq, from_legal_u64, to_legal_u64, *, elo_idx=None)` now returns `(from_logits, from_mask, to_logits, to_mask, value_logits)` (value appended — encode-once).

- [ ] **Step 1: Write the failing test**

`tests/style_policy/test_model_value.py`:

```python
import torch
from style_policy.model import BasePolicy

CFG = dict(d_model=64, n_layers=2, nhead=4, dim_feedforward=128, dropout=0.0,
           head_hidden=32, elo_dim=8, n_elo_buckets=40)

def _packed(b=3):
    t = torch.zeros(b, 34, dtype=torch.uint8)
    t[:, 32] = 1  # meta: white to move (bit 0)
    return t

def test_forward_value_shape_and_policy_returns_value():
    m = BasePolicy.from_config(CFG).eval()
    pk = _packed()
    elo = torch.tensor([10, 12, 14])
    v = m.forward_value(pk, elo_idx=elo)
    assert v.shape == (3, 3)
    fl, fm, tl, tm, v2 = m.forward_policy(pk, torch.zeros(3, dtype=torch.long),
                                          torch.ones(3, dtype=torch.int64), torch.ones(3, dtype=torch.int64),
                                          elo_idx=elo)
    assert v2.shape == (3, 3)
    assert torch.allclose(v, v2, atol=1e-5)  # same value from both paths (encode-once)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 pytest tests/style_policy/test_model_value.py -q'`
Expected: FAIL — `forward_value` missing / `forward_policy` returns 4 values.

- [ ] **Step 3: Implement**

In `style_policy/model.py`:

(a) import + constructor:

```python
from style_policy.value_head import WDLHead
```

Change `__init__` to accept and store `value_head`:

```python
    def __init__(self, encoder, from_head, to_head, promo_head, value_head):
        super().__init__()
        self.encoder = encoder
        self.from_head = from_head
        self.to_head = to_head
        self.promo_head = promo_head
        self.value_head = value_head
```

(b) `from_config` — build and pass the value head:

```python
        return cls(enc,
                   FromHead(d_model=d, hidden=h, elo_dim=elo_dim, n_elo_buckets=n_elo),
                   ToHead(d_model=d, hidden=h, elo_dim=elo_dim, n_elo_buckets=n_elo),
                   PromotionHead(d_model=d),
                   WDLHead(d_model=d, hidden=h, elo_dim=elo_dim, n_elo_buckets=n_elo))
```

(c) add `forward_value` and extend `forward_policy` (uses the `cls` the encoder already returns):

```python
    def forward_value(self, packed_pre, *, elo_idx=None):
        cls, _ = self.encode(packed_pre)
        return self.value_head(cls, elo_idx=elo_idx)

    def forward_policy(self, packed_pre, from_sq, from_legal_u64, to_legal_u64, *, elo_idx=None):
        """Encode once; return (from_logits, from_mask, to_logits, to_mask, value_logits)."""
        cls, squares = self.encode(packed_pre)
        from_logits = self.from_head(squares, elo_idx=elo_idx)
        from_mask = u64_to_mask(from_legal_u64).to(from_logits.device)
        to_logits = self.to_head(squares, from_sq, elo_idx=elo_idx)
        to_mask = u64_to_mask(to_legal_u64).to(to_logits.device)
        value_logits = self.value_head(cls, elo_idx=elo_idx)
        return (from_logits.masked_fill(~from_mask, float("-inf")), from_mask,
                to_logits.masked_fill(~to_mask, float("-inf")), to_mask, value_logits)
```

(Leave `forward_from`/`forward_to` unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 pytest tests/style_policy/test_model_value.py tests/style_policy/test_model_forward.py -q'`
Expected: new test PASS. NOTE: `test_model_forward.py` may call `forward_policy` expecting 4 returns — if it fails on the new 5-tuple, that's a real breakage from this interface change; update those call sites to unpack 5 values (the to-be-fixed reviewer item). The training-loop call site is updated in Task 8.

- [ ] **Step 5: Commit**

```bash
git add style_policy/model.py tests/style_policy/test_model_value.py
git commit -m "feat: BasePolicy WDL value head (forward_value + value from forward_policy)"
```

---

### Task 7: `wdl_ce` loss + accuracy metric

**Files:**
- Modify: `style_policy/loss.py`
- Test: `tests/style_policy/test_wdl_loss.py`

**Interfaces:**
- Produces: `wdl_ce(value_logits: (B,3), result: (B,)) -> scalar tensor` (mean 3-class cross-entropy). `wdl_accuracy(value_logits, result) -> float` (argmax == result).

- [ ] **Step 1: Write the failing test**

`tests/style_policy/test_wdl_loss.py`:

```python
import math, torch
from style_policy.loss import wdl_ce, wdl_accuracy

def test_wdl_ce_uniform_is_log3():
    logits = torch.zeros(4, 3)  # uniform -> CE = ln(3)
    target = torch.tensor([0, 1, 2, 1])
    assert abs(wdl_ce(logits, target).item() - math.log(3)) < 1e-5

def test_wdl_accuracy():
    logits = torch.tensor([[9.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0], [9.0, 0, 0]])
    target = torch.tensor([0, 1, 2, 2])
    assert abs(wdl_accuracy(logits, target) - 0.75) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 pytest tests/style_policy/test_wdl_loss.py -q'`
Expected: FAIL — import error.

- [ ] **Step 3: Implement**

Add to `style_policy/loss.py`:

```python
def wdl_ce(value_logits: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
    """3-class cross-entropy of WDL logits (order loss/draw/win) vs realized result label."""
    return F.cross_entropy(value_logits, result.long())


def wdl_accuracy(value_logits: torch.Tensor, result: torch.Tensor) -> float:
    return (value_logits.argmax(dim=-1) == result.long()).float().mean().item()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 pytest tests/style_policy/test_wdl_loss.py -q'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add style_policy/loss.py tests/style_policy/test_wdl_loss.py
git commit -m "feat: wdl_ce + wdl_accuracy"
```

---

### Task 8: Joint loss in the training loop + config weight

**Files:**
- Modify: `style_policy/training_loop.py`
- Create: `style_policy/model_configs/wdl_16M.yaml`, `style_policy/model_configs/tiny_smoke_wdl.yaml`
- Test: `tests/style_policy/test_training_value_smoke.py`

**Interfaces:**
- Consumes: `forward_policy` (5-tuple, Task 6), `wdl_ce`/`wdl_accuracy` (Task 7), `PackedMoveDataset` with `result` (Task 4).
- Produces: `_step_loss` returns `(total_loss, metrics)` where `total = from_ce + to_ce + λ·wdl_ce` (`λ` = `stage.get("value_loss_weight", 1.0)`), and `metrics` includes `wdl_ce`/`wdl_acc`. `_validate` aggregates `val/wdl_ce`, `val/wdl_acc`.

- [ ] **Step 1: Write the failing smoke test**

`tests/style_policy/test_training_value_smoke.py`:

```python
import numpy as np, h5py, torch
from dataset_generation.hdf5_io import PackedBatchWriter
from style_policy.dataset import PackedMoveDataset
from style_policy.model import BasePolicy
from style_policy.training_loop import _step_loss

CFG = dict(d_model=32, n_layers=2, nhead=4, dim_feedforward=64, dropout=0.0,
           head_hidden=16, elo_dim=8, n_elo_buckets=40)

def _make(path, n=8):
    with PackedBatchWriter(path, batch_size=n) as w:
        for i in range(n):
            pre = np.zeros(34, np.uint8); pre[32] = 1
            w.append_row(packed_pre=pre, from_legal_u64=(1 << 1), to_legal_u64=(1 << 2),
                         from_sq=1, to_sq=2, promotion=0, elo_to_move=1500, opp_elo=1500, result=i % 3)

def test_step_loss_includes_value_term(tmp_path):
    p = tmp_path / "t.h5"; _make(p)
    ds = PackedMoveDataset(p)
    batch = PackedMoveDataset.collate([ds[i] for i in range(len(ds))])
    model = BasePolicy.from_config(CFG)
    loss, m = _step_loss(model, batch, "cpu", 40, 0.0)
    assert torch.isfinite(loss)
    assert "wdl_ce" in m and "wdl_acc" in m
    # with value_loss_weight default 1.0, total > from_ce + to_ce (value term is positive here)
    assert loss.item() > m["from_ce"] + m["to_ce"] - 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 pytest tests/style_policy/test_training_value_smoke.py -q'`
Expected: FAIL — `_step_loss` unpacks 4 from `forward_policy` (now 5) / no `wdl_ce` in metrics.

- [ ] **Step 3: Implement**

In `style_policy/training_loop.py`:

(a) import: add `wdl_ce, wdl_accuracy` to the loss import line:

```python
from style_policy.loss import masked_square_ce, top1_legal, joint_top1, wdl_ce, wdl_accuracy
```

(b) `_step_loss` — add `value_loss_weight` param (default 1.0), unpack 5-tuple, add value loss + metrics:

```python
def _step_loss(model, batch, device, n_elo, label_smoothing, value_loss_weight=1.0):
    packed = batch["packed_pre"].to(device)
    elo_idx = elo_to_bucket(batch["elo_to_move"], n_elo).to(device)
    from_logits, from_mask, to_logits, to_mask, value_logits = model.forward_policy(
        packed, batch["from_sq"].to(device),
        batch["from_legal_u64"].to(device), batch["to_legal_u64"].to(device),
        elo_idx=elo_idx)
    from_sq = batch["from_sq"].to(device)
    to_sq = batch["to_sq"].to(device)
    result = batch["result"].to(device)
    fl = masked_square_ce(from_logits, from_sq, from_mask, label_smoothing=label_smoothing)
    tl = masked_square_ce(to_logits, to_sq, to_mask, label_smoothing=label_smoothing)
    vl = wdl_ce(value_logits, result)
    total = fl + tl + value_loss_weight * vl
    metrics = {"from_ce": fl.item(), "to_ce": tl.item(), "wdl_ce": vl.item(),
               "from_top1": top1_legal(from_logits, from_sq, from_mask),
               "to_top1": top1_legal(to_logits, to_sq, to_mask),
               "full_top1": joint_top1(from_logits, from_sq, from_mask, to_logits, to_sq, to_mask),
               "wdl_acc": wdl_accuracy(value_logits, result)}
    return total, metrics
```

(c) thread `value_loss_weight` from the stage config at the two `_step_loss` call sites:
- training step (in `train_one_stage` loop): `loss, m = _step_loss(model, batch, device, n_elo, stage.get("label_smoothing", 0.0), stage.get("value_loss_weight", 1.0))`
- `_validate`: change its call to `_, m = _step_loss(model, batch, device, n_elo, 0.0, stage_vlw)` — pass `value_loss_weight` into `_validate` (add a param) from `train_one_stage` as `stage.get("value_loss_weight", 1.0)`; and extend the `tot` dict + returned keys to include `"wdl_ce"` and `"wdl_acc"`:

```python
@torch.no_grad()
def _validate(model, val_dl, device, n_elo, use_amp, amp_dtype, value_loss_weight=1.0) -> dict:
    was_training = model.training
    model.eval()
    tot = {"from_ce": 0.0, "to_ce": 0.0, "from_top1": 0.0, "to_top1": 0.0,
           "full_top1": 0.0, "wdl_ce": 0.0, "wdl_acc": 0.0}
    nb = 0
    for batch in val_dl:
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp and device == "cuda"):
            _, m = _step_loss(model, batch, device, n_elo, 0.0, value_loss_weight)
        for k in tot:
            tot[k] += m[k]
        nb += 1
    if was_training:
        model.train()
    nb = max(nb, 1)
    return {f"val/{k}": tot[k] / nb for k in tot}
```

Update `_validate` call sites in `train_one_stage` to pass `stage.get("value_loss_weight", 1.0)`. Add `train/wdl_ce`, `train/wdl_acc` to the W&B `run.log` dict in the training-step log block.

(d) Create `style_policy/model_configs/tiny_smoke_wdl.yaml` (mirror `tiny_smoke.yaml` but pointing `train_h5`/`val_h5` at a tiny packed-with-result file and adding `value_loss_weight: 1.0` under `defaults`). Create `style_policy/model_configs/wdl_16M.yaml` (copy of `base_16M.yaml` with `name: wdl_16M`, `checkpoint_dir: style_policy_checkpoints/wdl_16M`, `train_h5: /mnt/eloquence_bulk/databases/wdl_training_16M.h5`, `val_h5: /mnt/eloquence_bulk/databases/wdl_validation_1M.h5`, and `value_loss_weight: 1.0` added under `defaults`).

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 pytest tests/style_policy/test_training_value_smoke.py -q'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add style_policy/training_loop.py style_policy/model_configs/wdl_16M.yaml style_policy/model_configs/tiny_smoke_wdl.yaml tests/style_policy/test_training_value_smoke.py
git commit -m "feat: joint policy+WDL training loss with value_loss_weight"
```

---

### Task 9: Recipes for the WDL regen (16M train + 1M val)

**Files:**
- Create: `dataset_generation/wdl_training_16M.yaml`, `dataset_generation/wdl_validation_1M.yaml`

**Interfaces:** none (data recipes). `wdl_training_16M` = the 16M recipe (the 32M recipe with `take_games: 200000`); `wdl_validation_1M` = the existing `validation_1M.yaml` content, renamed, output `wdl_validation_1M.h5`.

- [ ] **Step 1: Create the 16M training recipe**

`dataset_generation/wdl_training_16M.yaml` — copy `j3_training_32M.yaml` verbatim but set `name: wdl_training_16M` and change every `take_games: 400000` to `take_games: 200000` (10 strata × 200000 × samples_per_game 8 = 16,000,000).

- [ ] **Step 2: Create the validation recipe**

`dataset_generation/wdl_validation_1M.yaml` — copy `dataset_generation/validation_1M.yaml`, set `name: wdl_validation_1M`. (Confirm it draws from a different month/seed than training to avoid leakage — match whatever `validation_1M.yaml` already does; do not change its sampling, only the `name`.)

- [ ] **Step 3: Validate recipes parse**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && python -c "
from dataset_generation.recipe import Recipe
for n in [\"wdl_training_16M\",\"wdl_validation_1M\"]:
    r=Recipe.load(f\"dataset_generation/{n}.yaml\"); print(n, r.target_sample_rows())
"'`
Expected: prints `wdl_training_16M 16000000` and the validation row target without error.

- [ ] **Step 4: Commit**

```bash
git add dataset_generation/wdl_training_16M.yaml dataset_generation/wdl_validation_1M.yaml
git commit -m "feat: WDL regen recipes (16M train + 1M val)"
```

---

### Task 10: WDL evaluation script

**Files:**
- Create: `scripts/eval_wdl.py`
- Test: `tests/style_policy/test_eval_wdl.py`

**Interfaces:**
- Produces: `evaluate(checkpoint_path, val_h5, *, device="cpu", sample_n=None) -> dict` returning `{"wdl_logloss", "wdl_acc", "prior_logloss", "full_top1", "n"}`. `prior_logloss` = log-loss of the per-elo-bucket marginal W/D/L base rate (the baseline the head must beat). Uses `BasePolicy.forward_policy` for both value and policy metrics.

- [ ] **Step 1: Write the failing test**

`tests/style_policy/test_eval_wdl.py`:

```python
import numpy as np, torch
from dataset_generation.hdf5_io import PackedBatchWriter
from style_policy.model import BasePolicy
from scripts.eval_wdl import prior_logloss_from_results

def test_prior_logloss_matches_entropy():
    # results 0/1/2 each appearing equally -> prior is uniform -> logloss = ln(3)
    res = np.array([0, 1, 2] * 10)
    import math
    assert abs(prior_logloss_from_results(res) - math.log(3)) < 1e-6
```

(The full `evaluate` needs a trained checkpoint + GPU; the unit-testable core is `prior_logloss_from_results`. The end-to-end run is exercised in Task 12.)

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 pytest tests/style_policy/test_eval_wdl.py -q'`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

`scripts/eval_wdl.py`:

```python
#!/usr/bin/env python3
"""Evaluate the WDL head: log-loss vs the per-elo-bucket prior, WDL accuracy, and policy
full-move top-1 (the joint policy metric) on a packed-with-result val set."""
from __future__ import annotations
import argparse, math
import numpy as np
import torch
from torch.utils.data import DataLoader
from style_policy.model import BasePolicy
from style_policy.dataset import PackedMoveDataset
from style_policy.model_spec import elo_to_bucket
from style_policy.loss import joint_top1, wdl_accuracy


def prior_logloss_from_results(results: np.ndarray) -> float:
    """Log-loss of always predicting the global marginal W/D/L distribution."""
    counts = np.bincount(results.astype(np.int64), minlength=3).astype(np.float64)
    p = counts / counts.sum()
    eps = 1e-12
    return float(-np.mean([math.log(max(p[int(r)], eps)) for r in results]))


@torch.no_grad()
def evaluate(checkpoint_path: str, val_h5: str, *, device: str = "cpu", sample_n=None) -> dict:
    ck = torch.load(checkpoint_path, map_location=device)
    model = BasePolicy.from_config(ck["architecture"]); model.load_state_dict(ck["model"]); model.to(device).eval()
    n_elo = int(ck["architecture"]["n_elo_buckets"])
    ds = PackedMoveDataset(val_h5, sample_n=sample_n)
    dl = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=PackedMoveDataset.collate)
    tot_ll = tot_acc = tot_top1 = 0.0
    nb = 0
    all_results = []
    for batch in dl:
        elo_idx = elo_to_bucket(batch["elo_to_move"], n_elo).to(device)
        fl, fm, tl, tm, vlog = model.forward_policy(
            batch["packed_pre"].to(device), batch["from_sq"].to(device),
            batch["from_legal_u64"].to(device), batch["to_legal_u64"].to(device), elo_idx=elo_idx)
        result = batch["result"].to(device)
        tot_ll += torch.nn.functional.cross_entropy(vlog, result.long()).item()
        tot_acc += wdl_accuracy(vlog, result)
        tot_top1 += joint_top1(fl, batch["from_sq"].to(device), fm, tl, batch["to_sq"].to(device), tm)
        all_results.append(batch["result"].numpy())
        nb += 1
    nb = max(nb, 1)
    results = np.concatenate(all_results)
    return {"wdl_logloss": tot_ll / nb, "wdl_acc": tot_acc / nb,
            "prior_logloss": prior_logloss_from_results(results),
            "full_top1": tot_top1 / nb, "n": int(len(results))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-h5", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sample-n", type=int, default=200000)
    args = ap.parse_args()
    print(evaluate(args.checkpoint, args.val_h5, device=args.device, sample_n=args.sample_n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 pytest tests/style_policy/test_eval_wdl.py -q'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_wdl.py tests/style_policy/test_eval_wdl.py
git commit -m "feat: WDL eval (log-loss vs prior + policy full_top1)"
```

---

### Task 11: Build the 16M + 1M WDL datasets (controller-run; long CPU job)

**Files:** none (produces `/mnt/eloquence_bulk/databases/wdl_training_16M.h5`, `wdl_validation_1M.h5`).

This is NOT a TDD subagent task — it runs the pipeline built in Tasks 1–3 on real data. The controller launches and monitors it (it is long; consider sharded/background as the original 64M build was).

- [ ] **Step 1: Build the validation set (smaller; validates the pipeline first)**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 python -m dataset_generation build --recipe dataset_generation/wdl_validation_1M.yaml --data-dir /mnt/eloquence_bulk/databases --output-dir /mnt/eloquence_bulk/databases'`
Expected: writes `wdl_validation_1M.h5`; row count == recipe target.

- [ ] **Step 2: Sanity-check the built file**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && python -c "
import h5py, numpy as np
f=h5py.File(\"/mnt/eloquence_bulk/databases/wdl_validation_1M.h5\",\"r\")
print({k:(f[k].shape,str(f[k].dtype)) for k in f.keys()})
print(\"result dist\", np.bincount(f[\"result\"][:100000], minlength=3))
"'`
Expected: columns include `packed_pre (N,34)`, `result (int8)`, `opp_elo (int16)`; result distribution spans {0,1,2}.

- [ ] **Step 3: Build the 16M training set**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 python -m dataset_generation build --recipe dataset_generation/wdl_training_16M.yaml --data-dir /mnt/eloquence_bulk/databases --output-dir /mnt/eloquence_bulk/databases'`
Expected: writes `wdl_training_16M.h5` with `packed_pre.shape == (16000000, 34)`.

- [ ] **Step 4: Commit a build-log note**

Append the build commands, row counts, and result-distribution to `docs/DEVLOG.md`; commit.

---

### Task 12: Joint train at 16M + evaluate the gate (controller-run; long GPU job)

**Files:** none (produces `style_policy_checkpoints/wdl_16M/...`).

- [ ] **Step 1: Re-evaluate the policy-only baseline on the new val set**

Run `scripts/eval_wdl.py` is value-aware; for the baseline use the existing policy-only model's full_top1 on the SAME val set. Since `base_16M_stage_1.pt` has no value head, evaluate its `full_top1` with the existing policy eval path (or a `--policy-only` flag). Record `base_16M` full_top1 on `wdl_validation_1M.h5`.

- [ ] **Step 2: Launch the joint 16M training run**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 python -m style_policy.train --config wdl_16M --device cuda'` (mirror the existing `train.py` CLI; `--resume` supported). Monitor via W&B (`train/wdl_ce`, `train/wdl_acc`, `val/*`). Long (~hours).

- [ ] **Step 3: Evaluate the gate**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && OMP_NUM_THREADS=6 python scripts/eval_wdl.py --checkpoint style_policy_checkpoints/wdl_16M/wdl_16M_stage_1.pt --val-h5 /mnt/eloquence_bulk/databases/wdl_validation_1M.h5 --device cuda'`
Gate: `wdl_logloss < prior_logloss` (WDL is real) AND `full_top1 >= base_16M full_top1 - 0.005` (policy not meaningfully regressed).

- [ ] **Step 4: Record results + go/no-go**

Append metrics, the gate verdict, and a calibration note to `docs/DEVLOG.md`; commit. If the gate passes → proceed to Task 13. If not → stop and report (tune `value_loss_weight`, or reconsider CLS vs mean-pool input).

---

### Task 13: Scale to 64M (controller-run; gated on Task 12)

**Files:**
- Create: `dataset_generation/wdl_training_64M.yaml` (copy `j3_training_64M.yaml`, `name: wdl_training_64M`), `style_policy/model_configs/wdl_64M.yaml` (copy `wdl_16M.yaml`, point at `wdl_training_64M.h5`, `name: wdl_64M`, `checkpoint_dir: style_policy_checkpoints/wdl_64M`).

- [ ] **Step 1: Build `wdl_training_64M.h5`** (same command as Task 11 Step 3 with the 64M recipe; sharded as the original 64M build if needed).
- [ ] **Step 2: Joint train at 64M** (`--config wdl_64M`). Monitor.
- [ ] **Step 3: Evaluate** on `wdl_validation_1M.h5`; compare WDL + full_top1 to `base_64M` and to the 16M WDL run. Append to `docs/DEVLOG.md`; commit.

---

## Self-Review

**Spec coverage:** label=realized-outcome-from-STM (Task 1); store both elos / condition on mover (Tasks 1,2,5,6 — `opp_elo` stored, head uses mover elo); regen with result (Tasks 1–3,9,11); CLS-token value head (Tasks 5,6); joint loss λ=1.0 (Tasks 7,8); 16M-then-64M (Tasks 11–13); validation gate WDL-beats-prior + policy-no-regress (Tasks 10,12); drop packed_post (Task 2 schema). All covered.

**Placeholder scan:** every code step has complete code; commands have expected output. Two explicit, justified deferrals: Task 3 notes adapting the test to `Recipe`'s real API if it differs (the recipe *schema* is fixed by the 32M example), and Tasks 11–13 are long runs (not TDD) with exact commands. Task 6 Step 4 flags that `test_model_forward.py` call sites may need the 5-tuple update — that is a real, expected consequence of the interface change, to be fixed in that task or as a review item.

**Type consistency:** `collect_candidate_positions` 6-tuple `(ply,stm,elo,opp_elo,result,move)` is produced in Task 1 and consumed in Task 3. `PackedBatchWriter.append_row` kwargs (Task 2) match the builder call (Task 3) and the dataset reader keys (Task 4). `forward_policy` 5-tuple (Task 6) matches `_step_loss` unpack (Task 8) and `eval_wdl` unpack (Task 10). `WDLHead(d_model,hidden,elo_dim,n_elo_buckets)` (Task 5) matches `from_config` (Task 6). `result` order loss/draw/win is consistent across Tasks 1, 7, and the head's 3 logits.
