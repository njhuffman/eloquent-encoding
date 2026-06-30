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
        finished = [(p, rc) for p in list(running) if (rc := p.poll()) is not None]
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
