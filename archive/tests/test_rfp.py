"""Tests for rfp HDF5 packing and model forward."""

from __future__ import annotations

import numpy as np
import torch

from gfp.model import FromSquareMlpHead
from jepa3.loss import masked_square_ce
from rfp.build_stream import _pack_deltas
from rfp.h5_io import RfpH5Writer, assert_rfp_h5, rfp_h5_attrs, rfp_h5_row_count
from rfp.model import ResidualFromPredictor


def test_pack_deltas_full_history() -> None:
    n, d = 4, 8
    z = np.random.randn(n + 1, d).astype(np.float32)
    dz, mask = _pack_deltas(z, history_len=n)
    assert dz.shape == (n, d)
    assert mask.sum() == n
    expected = (z[1:] - z[:-1]).astype(np.float16)
    np.testing.assert_allclose(dz.astype(np.float32), expected.astype(np.float32), rtol=1e-3)


def test_pack_deltas_padded() -> None:
    n, d = 8, 4
    z = np.random.randn(3, d).astype(np.float32)
    dz, mask = _pack_deltas(z, history_len=n)
    assert dz.shape == (n, d)
    assert int(mask.sum()) == 2
    rd = z[1:] - z[:-1]
    np.testing.assert_allclose(
        dz[-2:].astype(np.float32), rd.astype(np.float32), rtol=1e-3
    )


def test_rfp_h5_roundtrip(tmp_path: object) -> None:
    path = tmp_path / "t.h5"
    hl, dm = 3, 16
    with RfpH5Writer(path, history_len=hl, d_model=dm, batch_size=2) as w:
        dz = np.zeros((hl, dm), dtype=np.float16)
        zc = np.ones(dm, dtype=np.float16)
        hm = np.ones(hl, dtype=np.uint8)
        w.append_row(
            delta_z=dz,
            z_curr=zc,
            history_mask=hm,
            from_legal_u64=np.uint64(123),
            from_sq=5,
            elo_bucket=15,
        )
        w.append_row(
            delta_z=dz + 1,
            z_curr=zc + 2,
            history_mask=hm,
            from_legal_u64=np.uint64(456),
            from_sq=10,
            elo_bucket=-1,
        )

    assert_rfp_h5(path)
    assert rfp_h5_row_count(path) == 2
    assert rfp_h5_attrs(path) == (hl, dm)


def test_residual_model_forward_shapes() -> None:
    torch.manual_seed(0)
    b, n, d = 4, 5, 32
    gfp_head = FromSquareMlpHead(d, 64, 2)
    for p in gfp_head.parameters():
        p.requires_grad_(False)

    model = ResidualFromPredictor(
        gfp_head,
        history_len=n,
        d_model=d,
        mixer_dim=48,
        mixer_depth=1,
        mixer_tokens_mlp_dim=32,
        mixer_channels_mlp_dim=64,
        mixer_dropout=0.0,
        elo_num_buckets=10,
        elo_embed_dim=8,
        residual_hidden=64,
        residual_depth=2,
    )
    dz = torch.randn(b, n, d)
    zc = torch.randn(b, d)
    hm = torch.ones(b, n)
    elo = torch.tensor([5, -1, 5, 5], dtype=torch.long)
    total, gfp = model(dz, zc, hm, elo, elo_null_prob=0.5, train=True)
    assert total.shape == (b, 64)
    assert gfp.shape == (b, 64)


def test_ce_sum_matches_identity() -> None:
    """Same logits → same masked CE (sanity for gfp + residual training target)."""
    torch.manual_seed(1)
    b = 8
    logits_a = torch.randn(b, 64)
    logits_b = logits_a.clone()
    fs = torch.randint(0, 64, (b,))
    m = torch.ones(b, 64)
    la, _ = masked_square_ce(logits_a, fs, m, label_smoothing=0.0)
    lb, _ = masked_square_ce(logits_b, fs, m, label_smoothing=0.0)
    assert torch.allclose(la, lb)
