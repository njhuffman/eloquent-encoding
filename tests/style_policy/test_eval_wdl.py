import numpy as np, torch
from dataset_generation.hdf5_io import PackedBatchWriter
from style_policy.model import BasePolicy
from scripts.eval_wdl import prior_logloss_from_results

def test_prior_logloss_matches_entropy():
    # results 0/1/2 each appearing equally -> prior is uniform -> logloss = ln(3)
    res = np.array([0, 1, 2] * 10)
    import math
    assert abs(prior_logloss_from_results(res) - math.log(3)) < 1e-6
