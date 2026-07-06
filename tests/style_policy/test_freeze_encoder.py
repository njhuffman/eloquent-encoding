"""Structural tests for encoder warm-start (init_encoder_from) + freeze_encoder.
CPU-only, tiny model, no data / no real training -- exercises the helper functions directly."""
import torch
from style_policy.multiband_policy import MultiBandPolicy
from style_policy.multiband_train import (
    _load_encoder_weights, _freeze_encoder, _trainable_params,
)

ARCH = {"d_model": 16, "n_layers": 1, "nhead": 2, "dim_feedforward": 32,
        "dropout": 0.0, "head_hidden": 8, "elo_dim": 4, "n_elo_buckets": 10,
        "bands": [1000, 1100]}


def test_init_encoder_from_copies_only_encoder(tmp_path):
    src = MultiBandPolicy.from_config(ARCH)
    ckpt = tmp_path / "src.pt"
    torch.save({"model": src.state_dict(), "architecture": ARCH}, ckpt)

    dst = MultiBandPolicy.from_config(ARCH)  # independent fresh init
    # encoders start out different (independent random init)
    enc_key = next(iter(dict(src.encoder.named_parameters())))
    src_enc = dict(src.encoder.named_parameters())
    dst_enc0 = dict(dst.encoder.named_parameters())
    assert not torch.allclose(src_enc[enc_key], dst_enc0[enc_key])

    n = _load_encoder_weights(dst, str(ckpt), "cpu")
    assert n == len(list(src.encoder.state_dict()))
    assert n > 0

    # every encoder param now matches the source checkpoint...
    dst_enc = dict(dst.encoder.named_parameters())
    for k in src_enc:
        assert torch.allclose(src_enc[k], dst_enc[k]), f"encoder param {k} not copied"

    # ...while heads stay at dst's fresh init (NOT copied from src)
    head_key = next(iter(dict(src.heads.named_parameters())))
    assert not torch.allclose(
        dict(src.heads.named_parameters())[head_key],
        dict(dst.heads.named_parameters())[head_key],
    ), "head params should remain fresh, not warm-started"


def test_freeze_encoder_and_optimizer_excludes_encoder():
    model = MultiBandPolicy.from_config(ARCH)
    n_frozen = _freeze_encoder(model)
    assert n_frozen == len(list(model.encoder.parameters()))

    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert any(p.requires_grad for p in model.heads.parameters())
    assert any(p.requires_grad for p in model.value_head.parameters())

    opt = torch.optim.AdamW(_trainable_params(model), lr=1e-3)
    opt_param_ids = {id(p) for grp in opt.param_groups for p in grp["params"]}
    enc_param_ids = {id(p) for p in model.encoder.parameters()}
    assert opt_param_ids.isdisjoint(enc_param_ids), "frozen encoder params leaked into optimizer"
    # heads/value head still optimized
    assert any(id(p) in opt_param_ids for p in model.heads.parameters())


def test_trainable_params_unchanged_when_nothing_frozen():
    """Backward-compat: with no freeze, _trainable_params == model.parameters() (same set + order)."""
    model = MultiBandPolicy.from_config(ARCH)
    got = _trainable_params(model)
    allp = list(model.parameters())
    assert len(got) == len(allp)
    assert all(a is b for a, b in zip(got, allp))
