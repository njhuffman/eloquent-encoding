"""train_multiband writes step-tagged snapshot checkpoints at configured snapshot_steps,
each a self-contained joint checkpoint (architecture + bands + model) loadable by
MultiBandPolicy.from_config -> load_state_dict, for the single-run scaling sweep."""
import torch
from style_policy.multiband_train import train_multiband
from style_policy.multiband_policy import MultiBandPolicy
from tests.style_policy.synth_h5 import write_synth_h5

ARCH = {"d_model": 32, "n_layers": 1, "nhead": 4, "dim_feedforward": 64, "dropout": 0.0,
        "head_hidden": 16, "elo_dim": 8, "n_elo_buckets": 40, "use_last_move": True, "n_history_ply": 2}


def test_snapshot_steps_written_and_loadable(tmp_path):
    h5 = write_synth_h5(tmp_path / "tr.h5", elos=[1000, 1500, 1900] * 64, with_hist=True)
    # n=256, batch=64 -> 4 steps; snapshot at steps 1 and 2.
    stage = {"compile": False, "use_amp": False, "amp_dtype": "bf16", "batch_size": 64,
             "dataloader_num_workers": 0, "weight_decay": 0.01, "max_gradient_norm": 1.0,
             "log_interval": 10, "val_interval": 0, "checkpoint_interval": 0,
             "lr_schedule": "constant", "warmup_steps": 0, "lr_min_frac": 0.0,
             "label_smoothing": 0.0, "value_loss_weight": 1.0,
             "last_move_dropout": 0.15, "history_dropout": "binary",
             "snapshot_steps": [1, 2],
             "sample": {"n": 256, "seed": 1}, "train": {"epochs": 1, "learning_rate": 3e-4}}
    spec = {"name": "snap", "checkpoint_dir": str(tmp_path / "ck"),
            "train_h5": str(h5), "architecture": ARCH, "stages": [stage]}
    train_multiband(spec, "cpu")

    for s in (1, 2):
        p = tmp_path / "ck" / f"snap.step{s}.pt"
        assert p.exists(), f"missing snapshot {p}"
        ck = torch.load(p)
        assert set(ck) >= {"architecture", "bands", "model"}
        m = MultiBandPolicy.from_config(ck["architecture"])
        m.load_state_dict(ck["model"])  # loads cleanly (strict) — proper self-contained ckpt
        assert len(ck["bands"]) == m.n_bands

    # a step NOT in snapshot_steps was not written
    assert not (tmp_path / "ck" / "snap.step3.pt").exists()
