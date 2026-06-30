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
    assert loads == [300, 300]              # LPT: {300} vs {200,100} — minimizes max load
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
    with pytest.raises(RuntimeError, match=r"shard build\(s\) failed.*shard.*\.log") as ei:
        run_parallel_mine(
            tmp_path / "par_smoke.yaml", data_dir, tmp_path / "out",
            shards_per_source=2, max_concurrency=2,
        )
    # "raise loudly": names a failing shard and points at its log directory.
    assert "par_smoke_shard" in str(ei.value)
