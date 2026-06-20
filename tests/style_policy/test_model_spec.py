import torch
from style_policy.model_spec import load_spec, elo_to_bucket


def test_elo_bucketing():
    elo = torch.tensor([-1, 0, 150, 2450, 99999])
    out = elo_to_bucket(elo, n_buckets=40)
    assert out[0].item() == 40 and out[1].item() == 40   # missing/zero → null
    assert out[2].item() == 1                            # 150 // 100
    assert out[3].item() == 24                            # 2450 // 100
    assert out[4].item() == 39                            # clamp


def test_load_tiny_smoke():
    spec = load_spec("tiny_smoke")
    assert spec["name"] == "tiny_smoke"
    assert spec["architecture"]["d_model"] <= 256
    assert len(spec["stages"]) >= 1
