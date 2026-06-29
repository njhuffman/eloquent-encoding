import numpy as np
from style_policy.dataset import PackedMoveDataset
from tests.style_policy.synth_h5 import write_synth_h5

def test_band_filter_selects_only_in_band(tmp_path):
    p = write_synth_h5(tmp_path / "d.h5", elos=[950, 1000, 1050, 1899, 1900, 1999, 2000])
    ds = PackedMoveDataset(str(p), band=(1900, 2000))
    import h5py
    with h5py.File(str(p), "r") as f:
        elo = f["elo_to_move"][:]
    assert len(ds) == 2
    assert all(1900 <= int(elo[i]) < 2000 for i in ds.indices)

def test_band_filter_subsamples_within_band(tmp_path):
    p = write_synth_h5(tmp_path / "d.h5", elos=[1900]*100 + [1000]*100)
    ds = PackedMoveDataset(str(p), band=(1900, 2000), sample_n=10, seed=1)
    assert len(ds) == 10
