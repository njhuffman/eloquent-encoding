import torch, h5py
from style_policy.model import BasePolicy

H5 = "/mnt/eloquence_bulk/databases/j3_training_1M.h5"
CFG = {"d_model": 64, "n_layers": 2, "nhead": 4, "dim_feedforward": 128, "dropout": 0.0,
       "head_hidden": 64, "elo_dim": 8, "n_elo_buckets": 40}


def _batch(n=4):
    with h5py.File(H5, "r") as f:
        return {
            "packed_pre": torch.from_numpy(f["packed_pre"][0:n].astype("uint8")),
            "from_sq": torch.from_numpy(f["from_sq"][0:n].astype("int64")),
            "from_legal_u64": torch.tensor([int(x) for x in f["from_legal_u64"][0:n]], dtype=torch.int64),
            "to_legal_u64": torch.tensor([int(x) for x in f["to_legal_u64"][0:n]], dtype=torch.int64),
        }


def test_forward_from_masks_illegal():
    m = BasePolicy.from_config(CFG).eval()
    b = _batch(4)
    with torch.no_grad():
        logits, mask = m.forward_from(b["packed_pre"], b["from_legal_u64"])
    assert logits.shape == (4, 64)
    # illegal squares are -inf
    assert torch.isinf(logits[~mask]).all()
    assert torch.isfinite(logits[mask]).all()


def test_forward_to_conditions_on_from():
    m = BasePolicy.from_config(CFG).eval()
    b = _batch(4)
    with torch.no_grad():
        logits, mask = m.forward_to(b["packed_pre"], b["from_sq"], b["to_legal_u64"])
    assert logits.shape == (4, 64)
    assert torch.isinf(logits[~mask]).all()
