import numpy as np, torch
from dataset_generation.hdf5_io import PackedBatchWriter
from style_policy.model import BasePolicy
from scripts.eval_wdl import prior_logloss_from_results

def test_prior_logloss_matches_entropy():
    # results 0/1/2 each appearing equally -> prior is uniform -> logloss = ln(3)
    res = np.array([0, 1, 2] * 10)
    import math
    assert abs(prior_logloss_from_results(res, np.zeros(30, dtype=np.int64)) - math.log(3)) < 1e-6

def test_per_bucket_prior_sharper_than_global():
    # bucket 0: all wins (result=2), bucket 1: all losses (result=0)
    # per-bucket prior perfectly predicts each row -> logloss ~ 0
    res = np.array([2] * 50 + [0] * 50)
    buckets = np.array([0] * 50 + [1] * 50)
    ll = prior_logloss_from_results(res, buckets)
    assert ll < 0.01, f"expected near-zero per-bucket logloss, got {ll}"
