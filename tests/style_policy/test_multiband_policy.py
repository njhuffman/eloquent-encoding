import torch
from style_policy.multiband_policy import MultiBandPolicy
from style_policy.packed_codec import PACKED_BOARD_LEN

ARCH = {"d_model": 32, "n_layers": 1, "nhead": 4, "dim_feedforward": 64,
        "dropout": 0.0, "head_hidden": 16, "elo_dim": 8, "n_elo_buckets": 40}

ARCH_HIST = dict(ARCH, use_last_move=True, n_history_ply=4)

def test_head_index_mapping_10_bands():
    m = MultiBandPolicy.from_config(ARCH)  # default 10 bands 1000-1900
    elo = torch.tensor([950, 1000, 1099, 1100, 1500, 1900, 1999, 2050])
    idx = m.head_index(elo)
    assert idx.tolist() == [0, 0, 0, 1, 5, 9, 9, 9]  # unchanged: 2000+ still clamps to top head


def test_head_index_mapping_12_bands():
    arch = dict(ARCH, bands=list(range(1000, 2200, 100)))  # 12 bands 1000-2100
    m = MultiBandPolicy.from_config(arch)
    assert m.n_bands == 12 and len(m.heads) == 12
    elo = torch.tensor([950, 1000, 1999, 2000, 2099, 2100, 2199, 2500])
    idx = m.head_index(elo)
    assert idx.tolist() == [0, 0, 9, 10, 10, 11, 11, 11]  # 2000-2099 -> head 10, 2100+ -> head 11

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


def test_train_multiband_hist_smoke(tmp_path):
    """Smoke test: use_last_move=True + hist batch columns + last_move_dropout > 0 trains without error."""
    from style_policy.multiband_train import train_multiband
    from tests.style_policy.synth_h5 import write_synth_h5
    h5 = write_synth_h5(tmp_path / "tr_hist.h5", elos=[1000, 1100, 1500, 1900] * 64, with_hist=True)
    stage = {"compile": False, "use_amp": False, "amp_dtype": "bf16", "batch_size": 64,
             "dataloader_num_workers": 0, "weight_decay": 0.01, "max_gradient_norm": 1.0,
             "log_interval": 10, "val_interval": 0, "checkpoint_interval": 0,
             "lr_schedule": "constant", "warmup_steps": 0, "lr_min_frac": 0.0,
             "label_smoothing": 0.0, "value_loss_weight": 1.0, "last_move_dropout": 0.25,
             "sample": {"n": 256, "seed": 1}, "train": {"epochs": 1, "learning_rate": 3e-4}}
    spec = {"name": "mb_hist_test", "checkpoint_dir": str(tmp_path / "ck"),
            "train_h5": str(h5), "architecture": ARCH_HIST, "stages": [stage]}
    result = train_multiband(spec, "cpu")
    assert result["steps"] > 0
    assert (tmp_path / "ck" / "mb_hist_test.pt").exists()
