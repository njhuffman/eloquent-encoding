import torch
from style_policy.multiband_policy import MultiBandPolicy
from style_policy.packed_codec import PACKED_BOARD_LEN

ARCH = {"d_model": 32, "n_layers": 1, "nhead": 4, "dim_feedforward": 64,
        "dropout": 0.0, "head_hidden": 16, "elo_dim": 8, "n_elo_buckets": 40}

def test_head_index_mapping():
    elo = torch.tensor([950, 1000, 1099, 1100, 1500, 1900, 1999, 2050])
    idx = MultiBandPolicy.head_index(elo)
    assert idx.tolist() == [0, 0, 0, 1, 5, 9, 9, 9]

def test_build_and_forward():
    m = MultiBandPolicy.from_config(ARCH)
    assert m.n_bands == 10 and len(m.heads) == 10
    import numpy as np
    packed = torch.zeros(4, PACKED_BOARD_LEN, dtype=torch.uint8); packed[:, 33] = 255
    cls, squares = m.encode(packed)
    assert cls.shape == (4, 32) and squares.shape == (4, 64, 32)
    fl = m.heads[3].from_logits(squares)
    assert fl.shape == (4, 64)
    v = m.value_head(cls, elo_idx=torch.full((4,), 15, dtype=torch.long))
    assert v.shape == (4, 3)


def test_train_multiband_smoke_and_exports(tmp_path):
    import torch
    from style_policy.multiband_train import train_multiband
    from style_policy.model import BasePolicy
    from style_policy.band_head import BandHead, BandHeadBot
    from tests.style_policy.synth_h5 import write_synth_h5
    h5 = write_synth_h5(tmp_path / "tr.h5", elos=[1000, 1100, 1500, 1900] * 64)  # mixed bands
    arch = dict(ARCH)
    stage = {"compile": False, "use_amp": False, "amp_dtype": "bf16", "batch_size": 64,
             "dataloader_num_workers": 0, "weight_decay": 0.01, "max_gradient_norm": 1.0,
             "log_interval": 10, "val_interval": 0, "checkpoint_interval": 0,
             "lr_schedule": "constant", "warmup_steps": 0, "lr_min_frac": 0.0,
             "label_smoothing": 0.0, "value_loss_weight": 1.0,
             "sample": {"n": 256, "seed": 1}, "train": {"epochs": 1, "learning_rate": 3e-4}}
    spec = {"name": "mb_test", "checkpoint_dir": str(tmp_path / "ck"),
            "train_h5": str(h5), "architecture": arch, "stages": [stage]}
    train_multiband(spec, "cpu")
    ck_dir = tmp_path / "ck"
    # joint + encoder + per-band exports exist and load
    assert (ck_dir / "mb_test.pt").exists()
    enc = torch.load(ck_dir / "mb_test_encoder.pt")
    BasePolicy.from_config(enc["architecture"]).load_state_dict(enc["model"], strict=False)  # encoder loads
    head_file = ck_dir / "band_heads" / "mb_test_band_1500.pt"
    st = torch.load(head_file)
    BandHead(st["d_model"], st["hidden"]).load_state_dict(st["band_head"])  # clean head load
    import chess
    bot = BandHeadBot(str(head_file), device="cpu", seed=0)  # plays a legal move
    assert bot.choose_move(chess.Board()) in chess.Board().legal_moves
