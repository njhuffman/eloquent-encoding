# Stockfish eval sidecar — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score dataset positions with Stockfish (static NNUE + depth-8 search + WDL) into a resumable sidecar HDF5, without touching the canonical dataset.

**Architecture:** Pure helpers (parse/encode) + sidecar I/O + a multiprocessing CLI where each worker drives one `python-chess` engine (search) and one raw Stockfish subprocess (static `eval`). Results write to a sidecar h5 aligned to source rows by `row_index`, with a `done` mask for resumability.

**Tech Stack:** Python, python-chess 1.11.2 (`chess.engine`), Stockfish 17.1, h5py, numpy, multiprocessing, pytest.

## Global Constraints

- **Execution environment:** run Python in the container —
  `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && PYTHONPATH=. <cmd>'`. Inside the container use `pytest` (NOT `python -m pytest`); scripts run as `python -m scripts.eval_stockfish`. Run `git` on the HOST.
- Stockfish binary: `/usr/games/stockfish` (Stockfish 17.1). python-chess `chess.engine`.
- **All evals are STM-relative** (match the dataset's `result` column).
- `sf_wdl` order is **[loss, draw, win]** (matches the model WDL head loss=0/draw=1/win=2), permille.
- Constants: `CP_CLAMP = 32000` (cp clamp; mate cp = ±CP_CLAMP); `STATIC_NA = -32768` (static eval undefined, e.g. in check).
- `--workers` default **8** (leave cores free).
- Branch: `stockfish-eval-sidecar`.
- Git commit footer (verbatim, both lines):
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01VMxeVCfznS5H68W5SGyXFC`

---

## File Structure

- `dataset_generation/stockfish_eval.py` — CREATE: constants, pure helpers (`clamp_cp`, `parse_static_eval`, `score_to_cp_mate`), sidecar I/O (`select_rows`, `open_or_create_sidecar`, `pending_positions`, `write_records`), engine pieces (`StaticEvalEngine`, `eval_position`).
- `scripts/eval_stockfish.py` — CREATE: CLI + multiprocessing orchestration.
- `tests/dataset_generation/test_stockfish_eval.py` — CREATE: pure-helper tests, sidecar/resume tests, Stockfish-gated integration test.

---

## Task 1: Pure helpers (parse + score encoding)

**Files:**
- Create: `dataset_generation/stockfish_eval.py`
- Test: `tests/dataset_generation/test_stockfish_eval.py`

**Interfaces:**
- Produces: `CP_CLAMP=32000`, `STATIC_NA=-32768`; `clamp_cp(cp:int)->int`; `parse_static_eval(text:str)->int|None` (cp from a Stockfish eval line; None if "none"/in-check); `score_to_cp_mate(score)->tuple[int,int]` (`score` is a python-chess `Score`; returns `(cp, mate)` where mate is signed UCI mate-in-N moves, 0 if none; mate → cp `±CP_CLAMP`).

- [ ] **Step 1: Write the failing test**

Create `tests/dataset_generation/test_stockfish_eval.py`:

```python
from chess.engine import Cp, Mate
from dataset_generation.stockfish_eval import (
    CP_CLAMP, STATIC_NA, clamp_cp, parse_static_eval, score_to_cp_mate,
)

def test_clamp_cp():
    assert clamp_cp(120) == 120
    assert clamp_cp(50000) == CP_CLAMP
    assert clamp_cp(-50000) == -CP_CLAMP

def test_parse_static_eval_normal():
    assert parse_static_eval("NNUE evaluation        +0.49 (white side)") == 49
    assert parse_static_eval("NNUE evaluation        -1.23 (white side)") == -123
    assert parse_static_eval("Final evaluation: +0.00 (white side)") == 0

def test_parse_static_eval_in_check():
    assert parse_static_eval("Final evaluation: none (in check)") is None

def test_score_to_cp_mate():
    assert score_to_cp_mate(Cp(120)) == (120, 0)
    assert score_to_cp_mate(Cp(50000)) == (CP_CLAMP, 0)   # clamped
    assert score_to_cp_mate(Mate(3)) == (CP_CLAMP, 3)
    assert score_to_cp_mate(Mate(-2)) == (-CP_CLAMP, -2)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && PYTHONPATH=. pytest tests/dataset_generation/test_stockfish_eval.py -x -q'`
Expected: FAIL — `dataset_generation.stockfish_eval` does not exist.

- [ ] **Step 3: Create `dataset_generation/stockfish_eval.py` with constants + pure helpers**

```python
"""Stockfish evaluation of dataset positions, written to a resumable sidecar HDF5.

All evals are from the side-to-move's perspective (matching the dataset `result` column).
"""
from __future__ import annotations
import re

CP_CLAMP = 32000        # centipawn clamp; a forced mate is stored as ±CP_CLAMP in the cp column
STATIC_NA = -32768      # sentinel: static eval undefined (e.g. side to move is in check)

_EVAL_RE = re.compile(r"(?:NNUE|Final) evaluation:?\s+([+-]?\d+\.\d+)")


def clamp_cp(cp: int) -> int:
    return max(-CP_CLAMP, min(CP_CLAMP, int(cp)))


def parse_static_eval(text: str) -> int | None:
    """Centipawns from a Stockfish `eval` final line, or None if undefined (in check)."""
    m = _EVAL_RE.search(text)
    if m:
        return round(float(m.group(1)) * 100)
    if "none" in text.lower():
        return None
    raise ValueError(f"could not parse static eval from: {text!r}")


def score_to_cp_mate(score) -> tuple[int, int]:
    """python-chess Score (STM-relative) -> (cp, mate). Mate -> cp ±CP_CLAMP; mate is signed UCI moves."""
    m = score.mate()
    if m is not None:
        return (CP_CLAMP if m > 0 else -CP_CLAMP, int(m))
    return (clamp_cp(score.score()), 0)
```

- [ ] **Step 4: Run the test**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && PYTHONPATH=. pytest tests/dataset_generation/test_stockfish_eval.py -x -q'`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add dataset_generation/stockfish_eval.py tests/dataset_generation/test_stockfish_eval.py
git commit -m "feat(sf-eval): pure helpers (clamp_cp, parse_static_eval, score_to_cp_mate)"
```

---

## Task 2: Sidecar I/O + resumability

**Files:**
- Modify: `dataset_generation/stockfish_eval.py`
- Test: `tests/dataset_generation/test_stockfish_eval.py`

**Interfaces:**
- Produces:
  - `select_rows(n_rows:int, sample:int|None, seed:int) -> np.ndarray` (sorted int64 source-row indices).
  - `open_or_create_sidecar(path:str, row_index:np.ndarray, attrs:dict) -> h5py.File` (r+; creates datasets `row_index`,`sf_static_cp`,`sf_cp`,`sf_mate`,`sf_wdl`,`done` sized N with sentinels, stores attrs; on reload validates `row_index` + key attrs match, else raises).
  - `pending_positions(f:h5py.File) -> np.ndarray` (sidecar indices where `done` is False).
  - `write_records(f:h5py.File, positions:list[int], records:list[dict]) -> None` (writes `static_cp`/`cp`/`mate`/`wdl` at each position, sets `done`, flushes). Each record dict has keys `cp, mate, static_cp, wdl` (`wdl` is a 3-tuple/list [loss,draw,win]).

- [ ] **Step 1: Write the failing test**

Append to `tests/dataset_generation/test_stockfish_eval.py`:

```python
import numpy as np
from dataset_generation.stockfish_eval import (
    select_rows, open_or_create_sidecar, pending_positions, write_records,
)

def _attrs():
    return {"source_h5": "x.h5", "source_n_rows": 100, "depth": 8, "sample_n": -1,
            "seed": 0, "perspective": "STM", "wdl_order": "loss,draw,win",
            "cp_clamp": CP_CLAMP, "stockfish_version": "test"}

def test_select_rows():
    assert np.array_equal(select_rows(5, None, 0), np.arange(5))
    s = select_rows(100, 10, 0)
    assert len(s) == 10 and len(set(s.tolist())) == 10 and list(s) == sorted(s)
    assert np.array_equal(select_rows(100, 10, 0), select_rows(100, 10, 0))  # seeded

def test_sidecar_create_pending_write_resume(tmp_path):
    p = str(tmp_path / "sc.h5")
    rows = np.arange(6, dtype=np.int64)
    f = open_or_create_sidecar(p, rows, _attrs())
    assert list(pending_positions(f)) == [0, 1, 2, 3, 4, 5]
    write_records(f, [0, 2], [
        {"cp": 30, "mate": 0, "static_cp": 25, "wdl": (100, 300, 600)},
        {"cp": -32000, "mate": -1, "static_cp": STATIC_NA, "wdl": (700, 200, 100)},
    ])
    assert list(pending_positions(f)) == [1, 3, 4, 5]
    assert f["sf_cp"][0] == 30 and f["sf_mate"][2] == -1
    assert list(f["sf_wdl"][0]) == [100, 300, 600]
    f.close()
    # reload: resumes (done preserved), attrs validated
    f2 = open_or_create_sidecar(p, rows, _attrs())
    assert list(pending_positions(f2)) == [1, 3, 4, 5]
    f2.close()

def test_sidecar_rejects_mismatch(tmp_path):
    p = str(tmp_path / "sc.h5")
    open_or_create_sidecar(p, np.arange(6, dtype=np.int64), _attrs()).close()
    try:
        open_or_create_sidecar(p, np.arange(7, dtype=np.int64), _attrs())  # different rows
        assert False, "expected mismatch error"
    except ValueError:
        pass
```

- [ ] **Step 2: Run it to verify it fails**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && PYTHONPATH=. pytest tests/dataset_generation/test_stockfish_eval.py -x -q'`
Expected: FAIL — `select_rows`/`open_or_create_sidecar` not defined.

- [ ] **Step 3: Add the sidecar I/O to `dataset_generation/stockfish_eval.py`**

Add these imports at the top (with the existing ones) and the functions below:

```python
import os
import numpy as np
import h5py
```

```python
def select_rows(n_rows: int, sample: int | None, seed: int) -> np.ndarray:
    if sample is None or sample >= n_rows:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=sample, replace=False)).astype(np.int64)


_MATCH_ATTRS = ("source_h5", "source_n_rows", "depth", "sample_n", "seed")


def open_or_create_sidecar(path: str, row_index: np.ndarray, attrs: dict) -> h5py.File:
    """Open an existing sidecar (validating alignment) or create a fresh one sized to row_index."""
    n = len(row_index)
    if os.path.exists(path):
        f = h5py.File(path, "r+")
        if f["row_index"].shape[0] != n or not np.array_equal(f["row_index"][:], row_index):
            f.close()
            raise ValueError("sidecar row_index mismatch — different file/sample/seed; refusing to resume")
        for k in _MATCH_ATTRS:
            if str(f.attrs.get(k)) != str(attrs[k]):
                f.close()
                raise ValueError(f"sidecar attr {k} mismatch ({f.attrs.get(k)} != {attrs[k]})")
        return f
    f = h5py.File(path, "w")
    f.create_dataset("row_index", data=row_index.astype(np.int64))
    f.create_dataset("sf_static_cp", shape=(n,), dtype="int16", fillvalue=STATIC_NA)
    f.create_dataset("sf_cp", shape=(n,), dtype="int16", fillvalue=0)
    f.create_dataset("sf_mate", shape=(n,), dtype="int8", fillvalue=0)
    f.create_dataset("sf_wdl", shape=(n, 3), dtype="int16", fillvalue=0)
    f.create_dataset("done", shape=(n,), dtype=bool, fillvalue=False)
    for k, v in attrs.items():
        f.attrs[k] = v
    f.flush()
    return f


def pending_positions(f: h5py.File) -> np.ndarray:
    return np.where(~f["done"][:])[0]


def write_records(f: h5py.File, positions, records) -> None:
    for pos, rec in zip(positions, records):
        f["sf_static_cp"][pos] = rec["static_cp"]
        f["sf_cp"][pos] = rec["cp"]
        f["sf_mate"][pos] = rec["mate"]
        f["sf_wdl"][pos] = rec["wdl"]
        f["done"][pos] = True
    f.flush()
```

- [ ] **Step 4: Run the test**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && PYTHONPATH=. pytest tests/dataset_generation/test_stockfish_eval.py -x -q'`
Expected: PASS (all Task-1 + Task-2 tests).

- [ ] **Step 5: Commit**

```bash
git add dataset_generation/stockfish_eval.py tests/dataset_generation/test_stockfish_eval.py
git commit -m "feat(sf-eval): resumable sidecar I/O (select/open/pending/write)"
```

---

## Task 3: Engine wrapper + eval_position + CLI + run

**Files:**
- Modify: `dataset_generation/stockfish_eval.py`
- Create: `scripts/eval_stockfish.py`
- Test: `tests/dataset_generation/test_stockfish_eval.py`

**Interfaces:**
- Consumes: Task 1 helpers, Task 2 sidecar I/O, `style_policy.board_encode.packed_to_board`.
- Produces:
  - `StaticEvalEngine(sf_path)` with `eval_cp(fen:str)->int|None` (static NNUE cp, STM-flipped by caller) and `close()`.
  - `eval_position(simple_engine, static_engine, board, depth) -> dict` with keys `cp, mate, static_cp, wdl` (STM-relative; `wdl` = [loss,draw,win] permille; terminal board → sentinels).
  - `scripts/eval_stockfish.py` CLI producing the sidecar.

- [ ] **Step 1: Write the failing (Stockfish-gated) integration test**

Append to `tests/dataset_generation/test_stockfish_eval.py`:

```python
import os, pytest, chess, chess.engine
from dataset_generation.stockfish_eval import StaticEvalEngine, eval_position

SF = "/usr/games/stockfish"

@pytest.mark.skipif(not os.path.exists(SF), reason="stockfish not installed")
def test_eval_position_integration():
    se = chess.engine.SimpleEngine.popen_uci(SF)
    se.configure({"Threads": 1, "Hash": 16, "UCI_ShowWDL": True})
    st = StaticEvalEngine(SF)
    try:
        # normal midgame: defined static eval, finite cp, wdl ~ permille
        r = eval_position(se, st, chess.Board(
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"), 8)
        assert -32000 <= r["cp"] <= 32000 and r["mate"] == 0
        assert r["static_cp"] != STATIC_NA
        assert abs(sum(r["wdl"]) - 1000) <= 2
        # white to move, mate in 1 (back-rank Re8#): positive STM mate
        r2 = eval_position(se, st, chess.Board("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1"), 8)
        assert r2["mate"] > 0
        # side to move in check: static eval undefined -> sentinel
        r3 = eval_position(se, st, chess.Board("4k3/8/4R3/8/8/8/8/4K3 b - - 0 1"), 8)
        assert r3["static_cp"] == STATIC_NA
    finally:
        se.quit(); st.close()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && PYTHONPATH=. pytest tests/dataset_generation/test_stockfish_eval.py::test_eval_position_integration -x -q'`
Expected: FAIL — `StaticEvalEngine`/`eval_position` not defined.

- [ ] **Step 3: Add `StaticEvalEngine` + `eval_position` to `dataset_generation/stockfish_eval.py`**

Add `import subprocess` at the top, then:

```python
class StaticEvalEngine:
    """A raw Stockfish process used only for the static `eval` command (python-chess lacks it)."""
    def __init__(self, sf_path: str):
        self.p = subprocess.Popen([sf_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, bufsize=1)
        self._send("uci")
        for line in self.p.stdout:           # drain until uciok
            if line.strip() == "uciok":
                break

    def _send(self, cmd: str) -> None:
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()

    def eval_cp(self, fen: str) -> int | None:
        self._send(f"position fen {fen}")
        self._send("eval")
        for line in self.p.stdout:
            s = line.strip()
            if s.startswith("NNUE evaluation") or s.startswith("Final evaluation"):
                return parse_static_eval(s)   # cp (white-relative) or None (in check)
        return None

    def close(self) -> None:
        try:
            self._send("quit")
            self.p.wait(timeout=2)
        except Exception:
            self.p.kill()


def eval_position(simple_engine, static_engine: StaticEvalEngine, board, depth: int) -> dict:
    """STM-relative record: {cp, mate, static_cp, wdl=[loss,draw,win]}. Terminal board -> sentinels."""
    import chess.engine
    if board.is_game_over():
        return {"cp": 0, "mate": 0, "static_cp": STATIC_NA, "wdl": (0, 0, 0)}
    info = simple_engine.analyse(board, chess.engine.Limit(depth=depth))
    cp, mate = score_to_cp_mate(info["score"].pov(board.turn))
    wdl_info = info.get("wdl")
    if wdl_info is not None:
        w = wdl_info.pov(board.turn)          # Wdl(wins, draws, losses), permille
        wdl = (int(w.losses), int(w.draws), int(w.wins))   # [loss, draw, win]
    else:
        wdl = (0, 0, 0)
    white_cp = static_engine.eval_cp(board.fen())          # white-relative, or None (in check)
    if white_cp is None:
        static_cp = STATIC_NA
    else:
        stm_cp = white_cp if board.turn == chess.WHITE else -white_cp
        static_cp = clamp_cp(stm_cp)
    return {"cp": cp, "mate": mate, "static_cp": static_cp, "wdl": wdl}
```

(Add `import chess` at module top for `chess.WHITE`; the `import chess.engine` inside the function is fine.)

- [ ] **Step 4: Run the integration test**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && PYTHONPATH=. pytest tests/dataset_generation/test_stockfish_eval.py -x -q'`
Expected: PASS (all pure + sidecar + the integration test).

- [ ] **Step 5: Create the CLI `scripts/eval_stockfish.py`**

```python
#!/usr/bin/env python3
"""Score dataset positions with Stockfish into a resumable sidecar HDF5.

Run: python -m scripts.eval_stockfish [--h5 ...] [--sample N] [--depth 8] [--workers 8]
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
import h5py
import chess, chess.engine
import multiprocessing as mp
from dataset_generation.stockfish_eval import (
    CP_CLAMP, select_rows, open_or_create_sidecar, pending_positions, write_records,
    StaticEvalEngine, eval_position,
)
from style_policy.board_encode import packed_to_board

_SE = _ST = _DEPTH = None


def _init(sf_path, hash_mb, depth):
    global _SE, _ST, _DEPTH
    _SE = chess.engine.SimpleEngine.popen_uci(sf_path)
    _SE.configure({"Threads": 1, "Hash": hash_mb, "UCI_ShowWDL": True})
    _ST = StaticEvalEngine(sf_path)
    _DEPTH = depth
    import atexit
    atexit.register(_cleanup)


def _cleanup():
    try: _SE.quit()
    except Exception: pass
    try: _ST.close()
    except Exception: pass


def _work(item):
    pos, fen = item
    return pos, eval_position(_SE, _ST, chess.Board(fen), _DEPTH)


def _sf_version(sf_path):
    e = chess.engine.SimpleEngine.popen_uci(sf_path)
    try:
        return e.id.get("name", "unknown")
    finally:
        e.quit()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="/mnt/eloquence_bulk/databases/wdl_validation_1M.h5")
    ap.add_argument("--out", default=None)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard-size", type=int, default=5000)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stockfish", default="/usr/games/stockfish")
    ap.add_argument("--hash", type=int, default=32)
    a = ap.parse_args()
    out = a.out or (os.path.splitext(a.h5)[0] + ".sf_eval.h5")

    with h5py.File(a.h5, "r") as f:
        n_rows = int(f["packed_pre"].shape[0])
        rows = select_rows(n_rows, a.sample, a.seed)
        packed = f["packed_pre"][rows]      # gathered in `rows` order (sorted index list)

    attrs = {"source_h5": a.h5, "source_n_rows": n_rows, "depth": a.depth,
             "sample_n": (a.sample if a.sample is not None else -1), "seed": a.seed,
             "perspective": "STM", "wdl_order": "loss,draw,win", "cp_clamp": CP_CLAMP,
             "stockfish_version": _sf_version(a.stockfish)}
    sc = open_or_create_sidecar(out, rows, attrs)
    try:
        pend = pending_positions(sc)
        print(f"{len(rows)} rows selected, {len(pend)} pending -> {out}", flush=True)
        items = [(int(pos), packed_to_board(np.asarray(packed[pos], np.uint8)).fen()) for pos in pend]
        t0 = time.time()
        buf_pos, buf_rec, n_done = [], [], 0
        with mp.Pool(a.workers, initializer=_init, initargs=(a.stockfish, a.hash, a.depth)) as pool:
            for pos, rec in pool.imap_unordered(_work, items, chunksize=16):
                buf_pos.append(pos); buf_rec.append(rec); n_done += 1
                if len(buf_pos) >= a.shard_size:
                    write_records(sc, buf_pos, buf_rec); buf_pos, buf_rec = [], []
                    print(f"  {n_done}/{len(items)} ({n_done/(time.time()-t0):.0f}/s)", flush=True)
            if buf_pos:
                write_records(sc, buf_pos, buf_rec)
        print(f"done {n_done} in {time.time()-t0:.0f}s", flush=True)
    finally:
        sc.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Commit the code**

```bash
git add dataset_generation/stockfish_eval.py scripts/eval_stockfish.py tests/dataset_generation/test_stockfish_eval.py
git commit -m "feat(sf-eval): engine wrapper, eval_position, and the CLI"
```

- [ ] **Step 7: Run it on the validation set + spot-check**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && PYTHONPATH=. python -m scripts.eval_stockfish --workers 8'`
Expected: prints `N rows selected, N pending -> /mnt/.../wdl_validation_1M.sf_eval.h5`, then `done N in <secs>s` (seconds for ~10k).
Then spot-check:
`docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && PYTHONPATH=. python -c "import h5py,numpy as np; f=h5py.File(\"/mnt/eloquence_bulk/databases/wdl_validation_1M.sf_eval.h5\"); print(\"done\", f[\"done\"][:].mean()); print(\"cp[:5]\", f[\"sf_cp\"][:5]); print(\"static[:5]\", f[\"sf_static_cp\"][:5]); print(\"wdl[0]\", list(f[\"sf_wdl\"][0]), \"sum\", int(sum(f[\"sf_wdl\"][0]))); print(\"static NA frac\", (f[\"sf_static_cp\"][:]==-32768).mean())"'`
Expected: `done 1.0`; cp/static values look like real centipawns; `wdl[0]` sums ≈ 1000; a small fraction static-NA (in-check positions). The `.sf_eval.h5` lives next to the source on `/mnt` (not committed to git).

- [ ] **Step 8: Commit (resume-marker / no code change)**

No code change in this step; the produced sidecar is a data artifact on `/mnt`, not committed. If the run surfaced a bug, fix it under TDD and amend Step 6's commit.

---

## Self-review notes

- **Spec coverage:** static NNUE + depth-8 search + WDL (T3 `eval_position`); STM perspective + clamps + sentinels (T1 constants/helpers, T3); sidecar schema + `row_index` + resumable `done` mask + attr/alignment validation (T2); ≤8-worker multiprocessing CLI + first run on the 10k val (T3); pure tests + sidecar tests + Stockfish-gated integration test (all tasks).
- **Naming consistency:** `CP_CLAMP`/`STATIC_NA`, `clamp_cp`/`parse_static_eval`/`score_to_cp_mate`, `select_rows`/`open_or_create_sidecar`/`pending_positions`/`write_records`, `StaticEvalEngine.eval_cp`/`eval_position`, record keys `cp,mate,static_cp,wdl` used identically across tasks and the CLI.
- **Known follow-ups (out of scope):** running the full 16M (`--sample`/different `--h5`); wiring the sidecar into probing analyses or training.
```
