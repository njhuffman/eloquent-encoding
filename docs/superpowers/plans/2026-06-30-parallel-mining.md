# Parallel, Training-Ready Data Mining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mine datasets across all cores (band-sharded `build` workers → seeded shuffle-merge) and read them sequentially at train time, so the held `wdl_history_64M` gen is ~8–10× faster and produces a globally-shuffled, training-ready file.

**Architecture:** A driver splits a recipe into per-band shard sub-recipes by **zeroing `take_games`** for all but each shard's assigned `(source, band)` cells — preserving `source_plan_index`/`stratum_index`/`stratum_seed` so each shard samples bit-identically to single-process. It launches throttled `python -m dataset_generation build` subprocesses, one shard h5 each. A shuffle-merge concatenates the shards and applies one seeded global permutation (per-dataset, to bound RAM) into one output h5. `PackedMoveDataset` gains a sequential-read mode so single-epoch training over the pre-shuffled file does zero random reads. No builder-internals change; the header prefilter (`pgn_prefilter.passes_header_prefilter`) already skips non-matching bands cheaply.

**Tech Stack:** Python, h5py, numpy, zstandard, PyTorch `Dataset`/`DataLoader`, `subprocess`, PyYAML, pytest.

## Global Constraints

- Tests run in container `1ec2b8ce64c8`: `docker exec -e PYTHONPATH=. 1ec2b8ce64c8 python -m pytest <path> -q`. There is **NO** `tests/style_policy/__init__.py` (adding one breaks namespace imports) — do not create package `__init__.py` under `tests/`.
- The sharding mechanism is **`take_games` zeroing only** — never reorder, add, or remove `source_plans`/`strata`; preserve every `stratum_seed`, `elo_min`, `elo_max`, `samples_per_game`, and the `master_seed`. Position must be identical to the input recipe so `builder._rng_for_game([master_seed, source_plan_index, stratum.stratum_seed, stratum_index, g])` is unchanged.
- A shard's non-zero cells must stay within a **single source_plan** (bounds duplicated decompression).
- A failed shard build must **raise loudly** naming the shard + its log path — never silently drop a band (a scarce band hitting `_ensure_strata_quotas_met` is a real failure to surface).
- Shuffle-merge uses **one seeded permutation** applied identically to every dataset (row alignment across columns is mandatory), processing **one dataset at a time** to bound peak RAM to ~2× the largest dataset.
- Sequential dataset mode defaults **off** (existing `rng.choice` random behavior preserved for un-shuffled files); back-compatible.
- Commit messages end with the required footer:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01VMxeVCfznS5H68W5SGyXFC
  ```

---

### Task 1: Band-sharded parallel mining driver

**Files:**
- Create: `scripts/mine_parallel.py`
- Test: `tests/dataset_generation/test_mine_parallel.py`

**Interfaces:**
- Consumes: recipe YAML schema (`name`, `master_seed`, `time_control`, `bucket_by`, `skip_opening_plies`, `exclude_single_legal_move`, `source_plans[].source`, `source_plans[].strata[].{elo_min,elo_max,take_games,samples_per_game,stratum_seed}`); the `python -m dataset_generation build --recipe R --data-dir D --output-dir O` CLI which writes `{O}/{name}.h5` and prints the path on stdout.
- Produces:
  - `make_shard_recipes(recipe: dict, shards_per_source: int) -> list[dict]` — sub-recipe dicts (deep copies, only `take_games` zeroed, `name` suffixed `_shardNN`).
  - `_balance_bins(weights: list[int], k: int) -> list[list[int]]` — greedy LPT partition of indices into `k` ascending-sorted groups.
  - `run_parallel_mine(recipe_path, data_dir, output_dir, *, shards_per_source: int, max_concurrency: int) -> list[Path]` — writes sub-recipes to `{output_dir}/_shards/`, launches throttled `build` subprocesses, returns shard h5 paths; raises `RuntimeError` on any shard failure.

- [ ] **Step 1: Write the failing tests**

```python
# tests/dataset_generation/test_mine_parallel.py
import h5py
import numpy as np
import zstandard
import pytest
from pathlib import Path
from scripts.mine_parallel import make_shard_recipes, _balance_bins, run_parallel_mine

_BASE = {
    "name": "demo", "master_seed": 1, "time_control": "600+0", "bucket_by": "white",
    "skip_opening_plies": 2, "exclude_single_legal_move": False,
    "source_plans": [
        {"source": "a.pgn.zst", "strata": [
            {"elo_min": 1000, "elo_max": 1099, "take_games": 100, "samples_per_game": 8, "stratum_seed": 11},
            {"elo_min": 1100, "elo_max": 1199, "take_games": 300, "samples_per_game": 8, "stratum_seed": 12},
        ]},
        {"source": "b.pgn.zst", "strata": [
            {"elo_min": 1000, "elo_max": 1099, "take_games": 200, "samples_per_game": 8, "stratum_seed": 21},
        ]},
    ],
}


def test_balance_bins_greedy_lpt():
    bins = _balance_bins([300, 100, 200], 2)
    loads = sorted(sum([300, 100, 200][i] for i in b) for b in bins)
    assert loads == [200, 400]              # {300,100} vs {200}
    assert all(b == sorted(b) for b in bins)  # ascending within bin


def test_make_shard_recipes_partitions_and_zeros():
    shards = make_shard_recipes(_BASE, shards_per_source=2)
    # names unique + suffixed
    names = [s["name"] for s in shards]
    assert len(names) == len(set(names)) and all(n.startswith("demo_shard") for n in names)
    orig = {(0, 0): 100, (0, 1): 300, (1, 0): 200}
    seen = {}
    for s in shards:
        # structure preserved exactly (zeroing only)
        assert [len(sp["strata"]) for sp in s["source_plans"]] == [2, 1]
        assert s["master_seed"] == 1
        assert s["source_plans"][0]["strata"][0]["stratum_seed"] == 11
        assert s["source_plans"][0]["strata"][1]["stratum_seed"] == 12
        assert s["source_plans"][1]["strata"][0]["stratum_seed"] == 21
        # a shard's work is within one source
        work_src = {si for si, sp in enumerate(s["source_plans"])
                    for st in sp["strata"] if st["take_games"] > 0}
        assert len(work_src) <= 1
        for si, sp in enumerate(s["source_plans"]):
            for ti, st in enumerate(sp["strata"]):
                if st["take_games"] > 0:
                    assert (si, ti) not in seen
                    seen[(si, ti)] = st["take_games"]
    assert seen == orig  # every cell covered exactly once, values intact


_GAME = """[Event "x"]
[White "a"]
[Black "b"]
[WhiteElo "{elo}"]
[BlackElo "{elo}"]
[TimeControl "600+0"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0

"""

_RECIPE = """name: par_smoke
master_seed: 1
time_control: 600+0
bucket_by: white
skip_opening_plies: 2
exclude_single_legal_move: false
source_plans:
  - source: smoke.pgn.zst
    strata:
      - {elo_min: 1500, elo_max: 1599, take_games: 2, samples_per_game: 3, stratum_seed: 1}
      - {elo_min: 1600, elo_max: 1699, take_games: 2, samples_per_game: 3, stratum_seed: 2}
"""


def test_run_parallel_mine_produces_shards(tmp_path):
    data_dir = tmp_path / "data"; data_dir.mkdir()
    raw = (_GAME.format(elo=1550) * 4 + _GAME.format(elo=1650) * 4).encode()
    with open(data_dir / "smoke.pgn.zst", "wb") as fh:
        fh.write(zstandard.ZstdCompressor().compress(raw))
    (tmp_path / "par_smoke.yaml").write_text(_RECIPE)
    out = run_parallel_mine(
        tmp_path / "par_smoke.yaml", data_dir, tmp_path / "out",
        shards_per_source=2, max_concurrency=2,
    )
    assert len(out) == 2 and all(p.exists() for p in out)
    total = 0
    for p in out:
        with h5py.File(p, "r") as f:
            total += int(f["packed_pre"].shape[0])
    # 2 bands x take_games=2 x samples_per_game=3 = 12 rows across shards
    assert total == 12


def test_run_parallel_mine_raises_on_failure(tmp_path):
    data_dir = tmp_path / "data"; data_dir.mkdir()
    # No games for band 1600 -> that shard fails its quota -> RuntimeError naming it.
    raw = (_GAME.format(elo=1550) * 4).encode()
    with open(data_dir / "smoke.pgn.zst", "wb") as fh:
        fh.write(zstandard.ZstdCompressor().compress(raw))
    (tmp_path / "par_smoke.yaml").write_text(_RECIPE)
    with pytest.raises(RuntimeError):
        run_parallel_mine(
            tmp_path / "par_smoke.yaml", data_dir, tmp_path / "out",
            shards_per_source=2, max_concurrency=2,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker exec -e PYTHONPATH=. 1ec2b8ce64c8 python -m pytest tests/dataset_generation/test_mine_parallel.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.mine_parallel'`.

- [ ] **Step 3: Implement the driver**

```python
# scripts/mine_parallel.py
"""Band-sharded parallel data mining: split a recipe into per-band shard sub-recipes
(via take_games zeroing, preserving sampling seeds), run `build` workers concurrently,
then shuffle-merge separately (see dataset_generation/shuffle_merge.py)."""
from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import time
from pathlib import Path

import yaml


def _balance_bins(weights: list[int], k: int) -> list[list[int]]:
    """Greedy longest-processing-time partition of indices [0..len(weights)) into k bins,
    balancing summed weight. Each returned bin is ascending-sorted."""
    if k <= 0:
        raise ValueError("k must be >= 1")
    bins: list[list[int]] = [[] for _ in range(k)]
    loads = [0] * k
    for i in sorted(range(len(weights)), key=lambda i: weights[i], reverse=True):
        j = min(range(k), key=lambda b: loads[b])
        bins[j].append(i)
        loads[j] += weights[i]
    return [sorted(b) for b in bins]


def make_shard_recipes(recipe: dict, shards_per_source: int) -> list[dict]:
    """One sub-recipe per (source, band-group). Each is a deep copy of `recipe` with
    take_games zeroed everywhere except its assigned cells; `name` gets a _shardNN suffix.
    Structure/seeds are untouched so per-band sampling is bit-identical to single-process."""
    shards: list[dict] = []
    shard_no = 0
    for si, plan in enumerate(recipe["source_plans"]):
        strata = plan["strata"]
        weights = [int(st["take_games"]) for st in strata]
        k = min(shards_per_source, len(strata))
        for grp in _balance_bins(weights, k):
            if not grp:
                continue
            sub = copy.deepcopy(recipe)
            sub["name"] = f'{recipe["name"]}_shard{shard_no:02d}'
            for sj, sp in enumerate(sub["source_plans"]):
                for ti, st in enumerate(sp["strata"]):
                    if not (sj == si and ti in grp):
                        st["take_games"] = 0
            shards.append(sub)
            shard_no += 1
    return shards


def run_parallel_mine(
    recipe_path,
    data_dir,
    output_dir,
    *,
    shards_per_source: int,
    max_concurrency: int,
) -> list[Path]:
    """Write shard sub-recipes to {output_dir}/_shards/, run `build` workers throttled to
    max_concurrency, and return the shard h5 paths. Raises RuntimeError if any shard fails."""
    recipe = yaml.safe_load(Path(recipe_path).read_text())
    shards = make_shard_recipes(recipe, shards_per_source)
    output_dir = Path(output_dir)
    shard_dir = output_dir / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    recipe_paths: list[Path] = []
    for sub in shards:
        p = shard_dir / f'{sub["name"]}.yaml'
        p.write_text(yaml.safe_dump(sub, sort_keys=False))
        recipe_paths.append(p)

    pending = list(recipe_paths)
    running: dict[subprocess.Popen, tuple[str, object]] = {}
    failures: list[str] = []
    outputs: list[Path] = []

    def launch(rp: Path) -> None:
        log = (shard_dir / f"{rp.stem}.log").open("wb")
        proc = subprocess.Popen(
            [sys.executable, "-m", "dataset_generation", "build",
             "--recipe", str(rp), "--data-dir", str(data_dir), "--output-dir", str(shard_dir)],
            stdout=log, stderr=subprocess.STDOUT,
        )
        running[proc] = (rp.stem, log)

    while pending or running:
        while pending and len(running) < max_concurrency:
            launch(pending.pop(0))
        finished = [(p, p.poll()) for p in list(running) if p.poll() is not None]
        if not finished:
            time.sleep(0.5)
            continue
        for proc, ret in finished:
            name, log = running.pop(proc)
            log.close()
            if ret != 0:
                failures.append(name)
            else:
                outputs.append(shard_dir / f"{name}.h5")

    if failures:
        raise RuntimeError(
            f"shard build(s) failed: {failures}; see logs in {shard_dir}/<shard>.log"
        )
    return sorted(outputs)


def main() -> int:
    ap = argparse.ArgumentParser(description="Band-sharded parallel mining (build workers).")
    ap.add_argument("--recipe", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--shards-per-source", type=int, default=12)
    ap.add_argument("--max-concurrency", type=int, default=12)
    a = ap.parse_args()
    outs = run_parallel_mine(
        a.recipe, a.data_dir, a.output_dir,
        shards_per_source=a.shards_per_source, max_concurrency=a.max_concurrency,
    )
    for p in outs:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker exec -e PYTHONPATH=. 1ec2b8ce64c8 python -m pytest tests/dataset_generation/test_mine_parallel.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/mine_parallel.py tests/dataset_generation/test_mine_parallel.py
git commit -m "feat(mining): band-sharded parallel build driver (take_games zeroing)"
```

---

### Task 2: Seeded shuffle-merge

**Files:**
- Create: `dataset_generation/shuffle_merge.py`
- Modify: `dataset_generation/__main__.py` (add a `merge` subcommand)
- Test: `tests/dataset_generation/test_shuffle_merge.py`

**Interfaces:**
- Consumes: shard h5 files (identical dataset names; the packed schema `packed_pre`, `from_legal_u64`, `to_legal_u64`, `from_sq`, `to_sq`, `promotion`, `elo_to_move`, `opp_elo`, `result`, `hist_from`, `hist_to`, `hist_cap`).
- Produces: `shuffle_merge(shard_paths: list, out_path, *, seed: int) -> Path` — one output h5 with `N = Σ rows`, every dataset permuted by the same seeded permutation.

- [ ] **Step 1: Write the failing tests**

```python
# tests/dataset_generation/test_shuffle_merge.py
import h5py
import numpy as np
import pytest
from pathlib import Path
from dataset_generation.shuffle_merge import shuffle_merge


def _fake_shard(path: Path, a_values: np.ndarray) -> None:
    """Two datasets with a known per-row relationship: b == a*10, vec == [a, a+1]."""
    with h5py.File(path, "w") as f:
        f.create_dataset("a", data=a_values.astype(np.int64))
        f.create_dataset("b", data=(a_values * 10).astype(np.int64))
        f.create_dataset("vec", data=np.stack([a_values, a_values + 1], axis=1).astype(np.int64))


def test_shuffle_merge_complete_aligned_deterministic(tmp_path):
    s0 = tmp_path / "s0.h5"; _fake_shard(s0, np.arange(0, 10))
    s1 = tmp_path / "s1.h5"; _fake_shard(s1, np.arange(10, 25))
    out = shuffle_merge([s0, s1], tmp_path / "merged.h5", seed=7)
    with h5py.File(out, "r") as f:
        a = f["a"][:]; b = f["b"][:]; vec = f["vec"][:]
    assert len(a) == 25                                   # completeness (count)
    assert sorted(a.tolist()) == list(range(25))          # completeness (values)
    assert np.array_equal(b, a * 10)                      # alignment a<->b
    assert np.array_equal(vec[:, 0], a) and np.array_equal(vec[:, 1], a + 1)  # alignment a<->vec
    assert not np.array_equal(a, np.sort(a))              # actually shuffled (seed 7, n=25)


def test_shuffle_merge_seed_reproducible(tmp_path):
    s0 = tmp_path / "s0.h5"; _fake_shard(s0, np.arange(0, 10))
    o1 = shuffle_merge([s0], tmp_path / "m1.h5", seed=3)
    o2 = shuffle_merge([s0], tmp_path / "m2.h5", seed=3)
    with h5py.File(o1, "r") as f1, h5py.File(o2, "r") as f2:
        assert np.array_equal(f1["a"][:], f2["a"][:])


def test_shuffle_merge_rejects_mismatched_datasets(tmp_path):
    s0 = tmp_path / "s0.h5"; _fake_shard(s0, np.arange(0, 5))
    with h5py.File(tmp_path / "s1.h5", "w") as f:
        f.create_dataset("a", data=np.arange(5).astype(np.int64))  # missing b, vec
    with pytest.raises(AssertionError):
        shuffle_merge([s0, tmp_path / "s1.h5"], tmp_path / "m.h5", seed=1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker exec -e PYTHONPATH=. 1ec2b8ce64c8 python -m pytest tests/dataset_generation/test_shuffle_merge.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dataset_generation.shuffle_merge'`.

- [ ] **Step 3: Implement shuffle_merge**

```python
# dataset_generation/shuffle_merge.py
"""Concatenate band-shard h5 files and apply one seeded global permutation, producing a
training-ready shuffled file. Processes one dataset at a time to bound peak RAM (~2x the
largest dataset). Row alignment across datasets is guaranteed by the single shared perm."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def shuffle_merge(shard_paths, out_path, *, seed: int) -> Path:
    shard_paths = [Path(p) for p in shard_paths]
    if not shard_paths:
        raise ValueError("no shard paths given")
    with h5py.File(shard_paths[0], "r") as f0:
        names = sorted(f0.keys())
        specs = {k: (f0[k].dtype, tuple(f0[k].shape[1:])) for k in names}

    lengths: list[int] = []
    for p in shard_paths:
        with h5py.File(p, "r") as f:
            assert sorted(f.keys()) == names, f"dataset name mismatch in {p}"
            lengths.append(int(f[names[0]].shape[0]))
    n = sum(lengths)
    perm = np.random.default_rng(seed).permutation(n)

    out_path = Path(out_path)
    with h5py.File(out_path, "w") as out:
        for k in names:
            dt, tail = specs[k]
            buf = np.empty((n, *tail), dtype=dt)
            off = 0
            for p, ln in zip(shard_paths, lengths):
                with h5py.File(p, "r") as f:
                    buf[off:off + ln] = f[k][:]
                off += ln
            out.create_dataset(k, data=buf[perm])
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Seeded shuffle-merge of band-shard h5 files.")
    ap.add_argument("--shards", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(shuffle_merge(a.shards, a.out, seed=a.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Add a `merge` subcommand to the package CLI**

In `dataset_generation/__main__.py`, after the `build` subparser block (before `args = parser.parse_args()`), add:

```python
    m = sub.add_parser("merge", help="Seeded shuffle-merge of band-shard h5 files")
    m.add_argument("--shards", type=Path, nargs="+", required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--seed", type=int, default=0)
```

Add the import at the top: `from dataset_generation.shuffle_merge import shuffle_merge`.
After the `if args.cmd == "build":` block returns, add:

```python
    if args.cmd == "merge":
        out = shuffle_merge(args.shards, args.out, seed=args.seed)
        print(out)
        return 0
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `docker exec -e PYTHONPATH=. 1ec2b8ce64c8 python -m pytest tests/dataset_generation/test_shuffle_merge.py -q`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add dataset_generation/shuffle_merge.py dataset_generation/__main__.py tests/dataset_generation/test_shuffle_merge.py
git commit -m "feat(mining): seeded shuffle-merge of band shards + merge CLI"
```

---

### Task 3: Sequential-read dataset path

**Files:**
- Modify: `style_policy/dataset.py` (`PackedMoveDataset.__init__`)
- Modify: `style_policy/training_loop.py` (`_make_loader` + its two call sites)
- Modify: `style_policy/multiband_train.py` (train/val loader construction)
- Test: `tests/style_policy/test_sequential_dataset.py`

**Interfaces:**
- Consumes: `PackedMoveDataset(h5_path, *, sample_n, seed, band)` (existing).
- Produces: `PackedMoveDataset(h5_path, *, sample_n, seed, band, sequential: bool = False)` — when `sequential` and `sample_n < len(pool)`, `self.indices = pool[:sample_n]` (first-N in file order) instead of `rng.choice`. Training reads `spec.get("presorted", False)` → passes `sequential=True` and sets the **train** DataLoader `shuffle=False`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/style_policy/test_sequential_dataset.py
import h5py
import numpy as np
from pathlib import Path
from style_policy.dataset import PackedMoveDataset


def _mk(path: Path, n: int) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("packed_pre", data=np.zeros((n, 34), np.uint8))
        f.create_dataset("elo_to_move", data=np.arange(1000, 1000 + n, dtype=np.int64))


def test_sequential_takes_first_n_in_order(tmp_path):
    p = tmp_path / "d.h5"; _mk(p, 100)
    ds = PackedMoveDataset(p, sample_n=10, seed=0, sequential=True)
    assert list(ds.indices) == list(range(10))


def test_random_mode_unchanged(tmp_path):
    p = tmp_path / "d.h5"; _mk(p, 100)
    ds = PackedMoveDataset(p, sample_n=10, seed=0, sequential=False)
    expected = np.sort(np.random.default_rng(0).choice(np.arange(100), size=10, replace=False))
    assert list(ds.indices) == list(expected)


def test_sequential_with_band_filter(tmp_path):
    p = tmp_path / "d.h5"; _mk(p, 100)  # elo 1000..1099 over indices 0..99
    ds = PackedMoveDataset(p, sample_n=5, seed=0, band=(1000, 1050), sequential=True)
    assert list(ds.indices) == list(range(5))  # band pool = 0..49; first 5 in order
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker exec -e PYTHONPATH=. 1ec2b8ce64c8 python -m pytest tests/style_policy/test_sequential_dataset.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'sequential'`.

- [ ] **Step 3: Add sequential mode to PackedMoveDataset**

In `style_policy/dataset.py`, change the signature and the subsample branch:

```python
    def __init__(self, h5_path: str | Path, *, sample_n: int | None = None, seed: int = 0,
                 band: tuple[int, int] | None = None, sequential: bool = False):
```

Replace the `if sample_n is not None and sample_n < len(pool):` block with:

```python
        if sample_n is not None and sample_n < len(pool):
            if sequential:
                # Pre-shuffled-on-disk file: take the first N in order (zero random reads).
                self.indices = pool[:sample_n]
            else:
                rng = np.random.default_rng(seed)
                self.indices = np.sort(rng.choice(pool, size=sample_n, replace=False))
        else:
            self.indices = pool  # nonzero()/arange() are already ascending
```

- [ ] **Step 4: Run dataset tests to verify they pass**

Run: `docker exec -e PYTHONPATH=. 1ec2b8ce64c8 python -m pytest tests/style_policy/test_sequential_dataset.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Wire the `presorted` flag into the training loops**

In `style_policy/training_loop.py`, change `_make_loader` (line ~79) to thread `sequential`:

```python
def _make_loader(h5: str, stage: dict, sample_n: int, seed: int, *, shuffle: bool, sequential: bool = False):
    ds = PackedMoveDataset(h5, sample_n=sample_n, seed=seed, sequential=sequential)
    dl = DataLoader(ds, batch_size=stage["batch_size"], shuffle=shuffle,
```

At the train call site (line ~128), read the flag once and apply it (presorted ⇒ sequential read + no DataLoader shuffle):

```python
    presorted = bool(spec.get("presorted", False))
    train_ds, train_dl = _make_loader(
        spec["train_h5"], stage, stage["sample"]["n"], stage["sample"]["seed"],
        shuffle=not presorted, sequential=presorted,
    )
```

Leave the val loader call (line ~131) unchanged (val already `shuffle=False`; sampling stays random for a stable metric).

In `style_policy/multiband_train.py` (lines ~104-105), apply the same to the train dataset/loader:

```python
    presorted = bool(spec.get("presorted", False))
    ds = PackedMoveDataset(spec["train_h5"], sample_n=stage["sample"]["n"],
                           seed=stage["sample"]["seed"], sequential=presorted)
    dl = DataLoader(ds, batch_size=stage["batch_size"], shuffle=not presorted,
```

Leave the multiband val loader (lines ~108-110) unchanged.

- [ ] **Step 6: Verify the wiring imports/loaders still construct (no regressions)**

Run: `docker exec -e PYTHONPATH=. 1ec2b8ce64c8 python -m pytest tests/style_policy/test_sequential_dataset.py tests/dataset_generation -q`
Expected: PASS (no import or signature errors in the training modules).

- [ ] **Step 7: Commit**

```bash
git add style_policy/dataset.py style_policy/training_loop.py style_policy/multiband_train.py tests/style_policy/test_sequential_dataset.py
git commit -m "feat(training): sequential-read dataset mode + presorted train flag"
```

---

## Self-Review

- **Spec coverage:** Part 1 (band-sharded workers) → Task 1; Part 2 (shuffle-merge) → Task 2; Part 3 (sequential-read dataset) → Task 3. The prefilter is pre-existing (no task, per spec). The held `wdl_history_64M` gen is out of scope (not a task).
- **Reproducibility constraints:** Task 1 zeroes `take_games` only and asserts structure/seeds preserved; Task 2 uses one seeded perm with alignment + determinism tests.
- **Back-compat:** Task 3 `sequential` defaults False; `presorted` defaults False — existing configs unaffected.
- **Type consistency:** `make_shard_recipes`/`_balance_bins`/`run_parallel_mine`/`shuffle_merge`/`sequential` signatures match between definition, tests, and call sites.
- **No placeholders:** every code step is complete.

## Post-implementation follow-ups (NOT tasks here)

- After landing: run the parallel mine + shuffle-merge for `wdl_history_64M` (HELD per the user — overnight gen, ≤14 cores; set `--max-concurrency`/`--shards-per-source` ≤ the core budget), then add `presorted: true` to `multiband_history_64M.yaml` to train on the shuffled file.
- The `2000–2199` strata at `take_games=400000` may be unfillable → a shard would raise; decide realized counts / lower those `take_games` before the real run.
