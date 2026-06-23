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
