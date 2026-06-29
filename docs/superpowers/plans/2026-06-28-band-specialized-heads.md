# Band-Specialized Heads (Frozen Encoder) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether per-band specialized policy heads on a frozen elo-agnostic encoder sharpen the elo diagonal versus the shared elo-conditioned head.

**Architecture:** Load a trained `BasePolicy`, freeze the encoder, and train an *unconditioned* (`elo_dim=0`) `FromHead`/`ToHead` on a single elo band's data (plain CE, legality-masked, reading frozen encoder features). Evaluate the band head's move-match across all bands and compare, on the same rows, to the shared head conditioned per band. Bias-free (each head is plain CE on its band) and keeps the encoder elo-agnostic.

**Tech Stack:** PyTorch, h5py, existing `style_policy` package (BasePolicy/encode, FromHead/ToHead, PackedMoveDataset, loss, legal_mask), run in the GPU container.

## Global Constraints

- The encoder is **frozen** during band-head training (`requires_grad_(False)`, `eval()`, encode under `no_grad`); no gradient reaches it. Encoder stays elo-agnostic.
- Band heads are **unconditioned**: `FromHead`/`ToHead` with `elo_dim=0` (no elo embedding).
- Use **`wdl_training_16M.h5`** (train) and **`wdl_validation_1M.h5`** (eval) — they load cleanly via `PackedMoveDataset` (have `result`/`opp_elo`); default frozen encoder = **`base_64M`** (`style_policy_checkpoints/base_64M/base_64M_stage_1.pt`).
- Bands are 100-wide over **1000–1900** (10 bands); binning matches `diagonal_check` (`band = clamp(floor(elo/100)*100, 1000, 1900)`), conditioning bucket via `model_spec.elo_to_bucket`.
- Eval metric mirrors `scripts/diagonal_check.py`: masked **argmax** move-match; full-move match = (pred_from == human_from) AND (pred_to == human_to).
- Unit tests are **hermetic** (synthetic tiny HDF5, CPU). The GPU experiment run (Task 5) is gated on the in-flight `wdl_16M_big` training freeing the GPU (~4h) — a concurrent GPU load OOMs the 4GB card.
- Reuse existing utilities; do not duplicate masking/loss logic.

---

### Task 1: Band filtering in `PackedMoveDataset`

**Files:**
- Modify: `style_policy/dataset.py`
- Create: `tests/style_policy/synth_h5.py` (shared test helper)
- Test: `tests/style_policy/test_band_filter.py`

**Interfaces:**
- Produces: `PackedMoveDataset(h5, *, sample_n=None, seed=0, band: tuple[int,int] | None = None)` — when `band=(lo,hi)`, restricts to rows with `lo <= elo_to_move < hi`, then optionally subsamples within the band.
- Consumes: nothing new.

- [ ] **Step 1: Shared synthetic-h5 helper**

Create `tests/style_policy/synth_h5.py`. Writes a minimal valid file in the WDL schema (empty-board packed rows so `packed_to_board_tensor` accepts them; all-legal masks):

```python
import h5py, numpy as np
from style_policy.packed_codec import PACKED_BOARD_LEN

def write_synth_h5(path, elos, *, seed=0):
    """Tiny WDL-schema h5: empty boards, all-legal masks, given elo_to_move list."""
    n = len(elos)
    rng = np.random.default_rng(seed)
    packed = np.zeros((n, PACKED_BOARD_LEN), dtype=np.uint8)
    packed[:, 33] = 255  # ep = none
    with h5py.File(path, "w") as f:
        f.create_dataset("packed_pre", data=packed)
        f.create_dataset("elo_to_move", data=np.asarray(elos, dtype=np.int16))
        f.create_dataset("opp_elo", data=np.full(n, 1500, dtype=np.int16))
        f.create_dataset("result", data=np.ones(n, dtype=np.int8))
        f.create_dataset("from_sq", data=rng.integers(0, 64, n).astype(np.uint8))
        f.create_dataset("to_sq", data=rng.integers(0, 64, n).astype(np.uint8))
        f.create_dataset("promotion", data=np.zeros(n, dtype=np.uint8))
        f.create_dataset("from_legal_u64", data=np.full(n, np.iinfo(np.uint64).max, dtype=np.uint64))
        f.create_dataset("to_legal_u64", data=np.full(n, np.iinfo(np.uint64).max, dtype=np.uint64))
    return path
```

- [ ] **Step 2: Write the failing test**

`tests/style_policy/test_band_filter.py`:

```python
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
```

- [ ] **Step 3: Run to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/style_policy/test_band_filter.py -q`
Expected: FAIL (`PackedMoveDataset` got unexpected kwarg `band`).

- [ ] **Step 4: Implement band filtering**

In `style_policy/dataset.py`, replace the `__init__` index-building block:

```python
def __init__(self, h5_path, *, sample_n=None, seed=0, band=None):
    self.path = str(h5_path)
    with h5py.File(self.path, "r") as f:
        n = int(f["packed_pre"].shape[0])
        if band is not None:
            elo = f["elo_to_move"][:]
            pool = np.nonzero((elo >= band[0]) & (elo < band[1]))[0]
        else:
            pool = np.arange(n)
    if sample_n is not None and sample_n < len(pool):
        rng = np.random.default_rng(seed)
        self.indices = np.sort(rng.choice(pool, size=sample_n, replace=False))
    else:
        self.indices = pool  # nonzero()/arange() are already ascending
    self._f = None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=. python -m pytest tests/style_policy/test_band_filter.py -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add style_policy/dataset.py tests/style_policy/synth_h5.py tests/style_policy/test_band_filter.py
git commit -m "feat(band-head): elo-band filtering in PackedMoveDataset"
```

---

### Task 2: `BandHead` module

**Files:**
- Create: `style_policy/band_head.py`
- Test: `tests/style_policy/test_band_head.py`

**Interfaces:**
- Produces: `BandHead(d_model: int, hidden: int)` with `.from_logits(squares)->(B,64)` and `.to_logits(squares, from_sq)->(B,64)`; uses `FromHead`/`ToHead` with `elo_dim=0`.

- [ ] **Step 1: Write the failing test**

`tests/style_policy/test_band_head.py`:

```python
import torch
from style_policy.band_head import BandHead

def test_band_head_shapes_and_unconditioned():
    h = BandHead(d_model=32, hidden=16)
    sq = torch.randn(4, 64, 32)
    fl = h.from_logits(sq)
    assert fl.shape == (4, 64) and torch.isfinite(fl).all()
    from_sq = torch.zeros(4, dtype=torch.long)
    tl = h.to_logits(sq, from_sq)
    assert tl.shape == (4, 64) and torch.isfinite(tl).all()
    # unconditioned: no elo embedding anywhere
    assert not any("elo_emb" in n for n, _ in h.named_parameters())
    assert sum(p.numel() for p in h.parameters()) > 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/style_policy/test_band_head.py -q`
Expected: FAIL (no module `style_policy.band_head`).

- [ ] **Step 3: Implement `BandHead`**

`style_policy/band_head.py`:

```python
"""Per-band specialized heads on a frozen encoder (elo-agnostic conditioning by hard band split)."""
from __future__ import annotations
import torch
import torch.nn as nn
from style_policy.policy_heads import FromHead, ToHead

class BandHead(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.from_head = FromHead(d_model=d_model, hidden=hidden, elo_dim=0)
        self.to_head = ToHead(d_model=d_model, hidden=hidden, elo_dim=0)

    def from_logits(self, squares: torch.Tensor) -> torch.Tensor:
        return self.from_head(squares)

    def to_logits(self, squares: torch.Tensor, from_sq: torch.Tensor) -> torch.Tensor:
        return self.to_head(squares, from_sq)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. python -m pytest tests/style_policy/test_band_head.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add style_policy/band_head.py tests/style_policy/test_band_head.py
git commit -m "feat(band-head): unconditioned BandHead module"
```

---

### Task 3: `train_band_head` (frozen encoder → train head → save)

**Files:**
- Modify: `style_policy/band_head.py`
- Modify: `tests/style_policy/test_band_head.py`

**Interfaces:**
- Consumes: `BasePolicy.encode`, `PackedMoveDataset(band=...)`, `loss.masked_square_ce`, `legal_mask.u64_to_mask`.
- Produces: `train_band_head(checkpoint, band, train_h5, *, device="cuda", steps=2000, batch_size=256, sample_n=None, lr=3e-4, label_smoothing=0.0, num_workers=4, seed=1, out=None) -> (BandHead, dict)`. Saves `{"band_head", "d_model", "hidden", "source_checkpoint", "band"}` to `out` if given.

- [ ] **Step 1: Write the failing smoke test**

Append to `tests/style_policy/test_band_head.py`:

```python
def test_train_band_head_smoke(tmp_path):
    import torch
    from style_policy.model import BasePolicy
    from style_policy.band_head import train_band_head, BandHead
    arch = {"d_model": 32, "n_layers": 1, "nhead": 4, "dim_feedforward": 64,
            "dropout": 0.0, "head_hidden": 32, "elo_dim": 8, "n_elo_buckets": 40}
    m = BasePolicy.from_config(arch)
    ckpt = tmp_path / "enc.pt"
    torch.save({"model": m.state_dict(), "architecture": arch}, ckpt)
    from tests.style_policy.synth_h5 import write_synth_h5
    h5 = write_synth_h5(tmp_path / "train.h5", elos=[1900]*256)
    out = tmp_path / "head.pt"
    head, meta = train_band_head(str(ckpt), 1900, str(h5), device="cpu",
                                 steps=5, batch_size=64, num_workers=0, out=str(out))
    assert meta["band"] == 1900 and meta["d_model"] == 32
    loaded = BandHead(meta["d_model"], meta["hidden"])
    loaded.load_state_dict(torch.load(out)["band_head"])  # clean load
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/style_policy/test_band_head.py::test_train_band_head_smoke -q`
Expected: FAIL (`train_band_head` not defined).

- [ ] **Step 3: Implement `train_band_head`**

Append to `style_policy/band_head.py`:

```python
from torch.utils.data import DataLoader
from style_policy.model import BasePolicy
from style_policy.dataset import PackedMoveDataset
from style_policy.loss import masked_square_ce
from style_policy.legal_mask import u64_to_mask

def train_band_head(checkpoint, band, train_h5, *, device="cuda", steps=2000,
                    batch_size=256, sample_n=None, lr=3e-4, label_smoothing=0.0,
                    num_workers=4, seed=1, out=None):
    ck = torch.load(checkpoint, map_location=device)
    arch = ck["architecture"]
    model = BasePolicy.from_config(arch); model.load_state_dict(ck["model"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d, h = int(arch["d_model"]), int(arch["head_hidden"])
    head = BandHead(d, h).to(device); head.train()
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    ds = PackedMoveDataset(train_h5, sample_n=sample_n, seed=seed, band=(band, band + 100))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                    collate_fn=PackedMoveDataset.collate)
    use_amp = device == "cuda"
    step = 0
    while step < steps:
        for batch in dl:
            if step >= steps:
                break
            packed = batch["packed_pre"].to(device)
            from_sq = batch["from_sq"].to(device); to_sq = batch["to_sq"].to(device)
            fmask = u64_to_mask(batch["from_legal_u64"].to(device))
            tmask = u64_to_mask(batch["to_legal_u64"].to(device))
            with torch.no_grad():
                _, squares = model.encode(packed)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                fl = head.from_logits(squares)
                tl = head.to_logits(squares, from_sq)
                loss = (masked_square_ce(fl, from_sq, fmask, label_smoothing=label_smoothing)
                        + masked_square_ce(tl, to_sq, tmask, label_smoothing=label_smoothing))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            step += 1
    meta = {"d_model": d, "hidden": h, "source_checkpoint": str(checkpoint), "band": int(band)}
    if out is not None:
        torch.save({"band_head": head.state_dict(), **meta}, out)
    return head, meta
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. python -m pytest tests/style_policy/test_band_head.py -q`
Expected: PASS (all band-head tests).

- [ ] **Step 5: Commit**

```bash
git add style_policy/band_head.py tests/style_policy/test_band_head.py
git commit -m "feat(band-head): train_band_head on a frozen encoder"
```

---

### Task 4: `eval_band_head_row` + CLIs

**Files:**
- Modify: `style_policy/band_head.py`
- Modify: `tests/style_policy/test_band_head.py`
- Create: `scripts/train_band_head.py`, `scripts/eval_band_head.py`

**Interfaces:**
- Produces: `eval_band_head_row(checkpoint, band_head, val_h5, bands, *, device="cuda", n=10000) -> dict[int, dict]` — per band `b`: `{"spec": <band_head move-match on band-b rows>, "shared": <shared head @bucket(b) move-match on the same rows>, "count": k}`. Mirrors `diagonal_check` masking.

- [ ] **Step 1: Write the failing test (forced-single-legal-move → 100%)**

Append to `tests/style_policy/test_band_head.py`:

```python
def test_eval_row_metric_on_forced_moves(tmp_path):
    import numpy as np, h5py, torch
    from style_policy.model import BasePolicy
    from style_policy.band_head import BandHead, eval_band_head_row
    from style_policy.packed_codec import PACKED_BOARD_LEN
    arch = {"d_model": 32, "n_layers": 1, "nhead": 4, "dim_feedforward": 64,
            "dropout": 0.0, "head_hidden": 32, "elo_dim": 8, "n_elo_buckets": 40}
    m = BasePolicy.from_config(arch); ckpt = tmp_path / "enc.pt"
    torch.save({"model": m.state_dict(), "architecture": arch}, ckpt)
    # one legal from-square and one legal to-square == the human move => any head scores 100%
    n = 8; vp = tmp_path / "val.h5"
    packed = np.zeros((n, PACKED_BOARD_LEN), np.uint8); packed[:, 33] = 255
    with h5py.File(vp, "w") as f:
        f["packed_pre"] = packed
        f["elo_to_move"] = np.full(n, 1900, np.int16)
        f["from_sq"] = np.full(n, 12, np.uint8); f["to_sq"] = np.full(n, 28, np.uint8)
        f["from_legal_u64"] = np.full(n, np.uint64(1) << np.uint64(12), np.uint64)
        f["to_legal_u64"] = np.full(n, np.uint64(1) << np.uint64(28), np.uint64)
        f["promotion"] = np.zeros(n, np.uint8); f["opp_elo"] = np.full(n, 1500, np.int16)
        f["result"] = np.ones(n, np.int8)
    head = BandHead(32, 32)
    rows = eval_band_head_row(str(ckpt), head, str(vp), [1900], device="cpu", n=n)
    assert rows[1900]["count"] == n
    assert rows[1900]["spec"] == 100.0 and rows[1900]["shared"] == 100.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/style_policy/test_band_head.py::test_eval_row_metric_on_forced_moves -q`
Expected: FAIL (`eval_band_head_row` not defined).

- [ ] **Step 3: Implement `eval_band_head_row`**

Append to `style_policy/band_head.py`:

```python
import numpy as np
import h5py
from style_policy.board_encode import packed_to_board, legal_from_u64, legal_to_u64
from style_policy.model_spec import elo_to_bucket
_NEG = float("-inf")

def _mask1(u64, dev):
    return u64_to_mask(torch.from_numpy(np.array([u64], dtype=np.uint64)).to(torch.int64)).to(dev)

@torch.no_grad()
def eval_band_head_row(checkpoint, band_head, val_h5, bands, *, device="cuda", n=10000):
    ck = torch.load(checkpoint, map_location=device)
    arch = ck["architecture"]; n_elo = int(arch["n_elo_buckets"])
    model = BasePolicy.from_config(arch); model.load_state_dict(ck["model"])
    model.to(device).eval()
    band_head = band_head.to(device).eval()
    with h5py.File(val_h5, "r") as f:
        m = min(n, f["packed_pre"].shape[0])
        packed = f["packed_pre"][:m]; hf = f["from_sq"][:m]; ht = f["to_sq"][:m]; elo = f["elo_to_move"][:m]
    out = {b: {"spec": 0, "shared": 0, "count": 0} for b in bands}
    for i in range(m):
        b = int(min(max(bands), max(min(bands), (int(elo[i]) // 100) * 100)))
        if b not in out:
            continue
        board = packed_to_board(np.asarray(packed[i], np.uint8))
        if board.is_game_over():
            continue
        out[b]["count"] += 1
        pk = torch.from_numpy(np.asarray(packed[i], np.uint8)[None]).to(device)
        _, squares = model.encode(pk)
        fmask = _mask1(legal_from_u64(board), device)
        bi = elo_to_bucket(torch.tensor([b]), n_elo).to(device)
        for tag, ffn, tfn in (
            ("spec", lambda s: band_head.from_logits(s), lambda s, pf: band_head.to_logits(s, pf)),
            ("shared", lambda s: model.from_head(s, elo_idx=bi), lambda s, pf: model.to_head(s, pf, elo_idx=bi)),
        ):
            pf = int(ffn(squares).masked_fill(~fmask, _NEG).argmax())
            tmask = _mask1(legal_to_u64(board, pf), device)
            pft = torch.tensor([pf], device=device)
            pt = int(tfn(squares, pft).masked_fill(~tmask, _NEG).argmax())
            if pf == int(hf[i]) and pt == int(ht[i]):
                out[b][tag] += 1
    return {b: {"spec": 100.0 * v["spec"] / v["count"] if v["count"] else 0.0,
                "shared": 100.0 * v["shared"] / v["count"] if v["count"] else 0.0,
                "count": v["count"]} for b, v in out.items()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. python -m pytest tests/style_policy/test_band_head.py -q`
Expected: PASS (all).

- [ ] **Step 5: Add the CLIs**

`scripts/train_band_head.py`:

```python
#!/usr/bin/env python3
"""Train a band-specialized head on a frozen encoder."""
import argparse
from style_policy.band_head import train_band_head

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="style_policy_checkpoints/base_64M/base_64M_stage_1.pt")
    ap.add_argument("--band", type=int, required=True)
    ap.add_argument("--train-h5", default="/mnt/eloquence_bulk/databases/wdl_training_16M.h5")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--sample-n", type=int, default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    _, meta = train_band_head(a.checkpoint, a.band, a.train_h5, device=a.device, steps=a.steps,
                              batch_size=a.batch_size, sample_n=a.sample_n, lr=a.lr,
                              label_smoothing=a.label_smoothing, out=a.out)
    print("saved", a.out, meta)

if __name__ == "__main__":
    raise SystemExit(main())
```

`scripts/eval_band_head.py`:

```python
#!/usr/bin/env python3
"""Evaluate a band head's move-match across bands vs the shared head, on the same rows."""
import argparse, torch
from style_policy.band_head import BandHead, eval_band_head_row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="style_policy_checkpoints/base_64M/base_64M_stage_1.pt")
    ap.add_argument("--head", required=True)
    ap.add_argument("--val-h5", default="/mnt/eloquence_bulk/databases/wdl_validation_1M.h5")
    ap.add_argument("--bands", type=int, nargs="+", default=list(range(1000, 2000, 100)))
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    st = torch.load(a.head, map_location=a.device)
    head = BandHead(st["d_model"], st["hidden"]); head.load_state_dict(st["band_head"])
    rows = eval_band_head_row(a.checkpoint, head, a.val_h5, a.bands, device=a.device, n=a.n)
    trained_band = st["band"]
    print(f"band head trained @ {trained_band}  (spec = this head, shared = baseline @ each band)")
    print(f"{'band':>6} {'count':>6} {'spec%':>7} {'shared%':>8} {'edge':>6}")
    for b in a.bands:
        r = rows[b]
        print(f"{b:>6} {r['count']:>6} {r['spec']:>7.1f} {r['shared']:>8.1f} {r['spec']-r['shared']:>+6.1f}")

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Commit**

```bash
git add style_policy/band_head.py tests/style_policy/test_band_head.py scripts/train_band_head.py scripts/eval_band_head.py
git commit -m "feat(band-head): eval_band_head_row + train/eval CLIs"
```

---

### Task 5: Run the band-1900 quick experiment (GPU, gated)

**Files:** none (execution + interpretation). **Precondition:** the in-flight `wdl_16M_big` training has finished (GPU free) — verify `nvidia-smi` shows the card idle before starting.

- [ ] **Step 1: Train the band-1900 head on `base_64M`'s frozen encoder**

Run (in container):
```bash
PYTHONPATH=. python -m scripts.train_band_head --band 1900 \
  --checkpoint style_policy_checkpoints/base_64M/base_64M_stage_1.pt \
  --train-h5 /mnt/eloquence_bulk/databases/wdl_training_16M.h5 \
  --steps 6000 --batch-size 256 --device cuda \
  --out style_policy_checkpoints/band_heads/base_64M_band_1900.pt
```
Expected: saves the head; trains in ~15–25 min (frozen-encoder forward + tiny head).

- [ ] **Step 2: Evaluate its diagonal row vs the shared head**

Run:
```bash
PYTHONPATH=. python -m scripts.eval_band_head \
  --checkpoint style_policy_checkpoints/base_64M/base_64M_stage_1.pt \
  --head style_policy_checkpoints/band_heads/base_64M_band_1900.pt \
  --val-h5 /mnt/eloquence_bulk/databases/wdl_validation_1M.h5 --device cuda
```

- [ ] **Step 3: Apply the decision criteria and record**

Promising if, on the same rows: (a) `spec%` at band 1900 **> `shared%`** at band 1900, AND (b) the `spec%` row is **peaked at 1900** (argmax over bands) with a larger 1900-vs-1000 drop than the shared head shows. Write the result (the row table + verdict) to memory `diagonal-findings.md`, noting whether to proceed to all-band heads / the joint-trained version.

---

### Task 6 (optional, follow-up): All-band specialized diagonal

**Files:** none new (reuse Task 4 tooling).

- [ ] Train band heads for all 10 bands (1000–1900), eval each across all bands, assemble the full specialized diagonal matrix, and compare "diagonal-is-best in X/10 bands" + mean edge to the shared-head baseline (2/10, ~+0.5%). Decide whether the joint shared-encoder + per-band-heads training (out of scope here) is worth building.

---

## Self-Review

- **Spec coverage:** band filtering (T1), unconditioned head (T2), frozen-encoder training (T3), cross-band eval + baseline + CLIs (T4), the band-1900 quick test + decision criteria (T5), optional full diagonal (T6) — all covered.
- **Type consistency:** `train_band_head` saves `{d_model, hidden, band, band_head}`; CLIs/eval reconstruct `BandHead(d_model, hidden)` and read `band` — consistent. `eval_band_head_row` returns `{band: {spec, shared, count}}`, used by the eval CLI.
- **Hermetic tests:** synthetic h5 helper avoids data/GPU dependence; GPU run isolated to T5 (gated).
- **Reuse:** encode/heads/loss/legal_mask/diagonal masking reused; no duplication.
