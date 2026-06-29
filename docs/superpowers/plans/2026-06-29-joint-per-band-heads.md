# Joint Per-Band Heads Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Train a shared elo-input-free encoder jointly with 10 per-band policy heads, then rate each head vs Maia2 to test whether the co-adapting encoder extends the strength ladder past the frozen ~1800 ceiling.

**Architecture:** New `MultiBandPolicy` (shared `BoardEncoder` + 10 `BandHead` + shared `WDLHead`); a `train_multiband` loop that routes each sample to its band's heads (size-weighted CE) plus shared value CE; exports per-band `BandHead` files + a `BasePolicy`-loadable encoder checkpoint so existing `rate_band_heads`/`eval_band_head_row` work unchanged.

**Tech Stack:** PyTorch, existing `style_policy` (BoardEncoder, BandHead, WDLHead, PackedMoveDataset, loss, legal_mask, model_spec), GPU container.

## Global Constraints

- Encoder takes **no elo input**; bands are selected by routing to a head (`head index = clamp((elo−1000)//100, 0, 9)`). This binning matches `diagonal_check`/`rate_band_heads`.
- Reuse existing modules: `BandHead` (from/to, `elo_dim=0`), `WDLHead`, `masked_square_ce`, `wdl_ce`, `u64_to_mask`, `elo_to_bucket`, `PackedMoveDataset`. No logic duplication.
- CUDA-gated speedups mirror `train_one_stage`: `torch.compile(encoder)`, fused AdamW, `set_float32_matmul_precision("high")`. Strip the `encoder._orig_mod.` prefix on save so the encoder checkpoint loads into a plain `BasePolicy`.
- Unit tests are hermetic (synthetic tiny HDF5 via `tests/style_policy/synth_h5.py`, CPU). The ~3h GPU run (Task 4) is gated on the GPU being free.
- Data: `wdl_training_16M.h5` / `wdl_validation_1M.h5`; arch 256/8 (~7M); baseline = existing `wdl_16M`.

---

### Task 1: `MultiBandPolicy` model

**Files:** Create `style_policy/multiband_policy.py`; Test `tests/style_policy/test_multiband_policy.py`

**Interfaces:**
- Produces: `MultiBandPolicy.from_config(cfg)` (`cfg` has d_model/n_layers/nhead/dim_feedforward/dropout/head_hidden/elo_dim/n_elo_buckets, optional `bands`); `.encode(packed)->(cls,squares)`; `.heads` (ModuleList of `BandHead`); `.value_head`; `.bands`, `.n_bands`; staticmethod `.head_index(elo)->LongTensor`.

- [ ] **Step 1: Write the failing test**

`tests/style_policy/test_multiband_policy.py`:
```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/style_policy/test_multiband_policy.py -q` → FAIL (no module).

- [ ] **Step 3: Implement**

`style_policy/multiband_policy.py`:
```python
"""MultiBandPolicy: shared elo-agnostic encoder + N per-band policy heads + shared value head."""
from __future__ import annotations
import torch
import torch.nn as nn
from style_policy.board_encoder import BoardEncoder
from style_policy.band_head import BandHead
from style_policy.value_head import WDLHead
from style_policy.packed_codec import packed_to_board_tensor

BANDS = list(range(1000, 2000, 100))  # 1000..1900


class MultiBandPolicy(nn.Module):
    def __init__(self, encoder, heads, value_head, bands=BANDS):
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleList(heads)
        self.value_head = value_head
        self.bands = list(bands)
        self.n_bands = len(self.bands)

    @classmethod
    def from_config(cls, cfg: dict) -> "MultiBandPolicy":
        d = int(cfg["d_model"]); h = int(cfg["head_hidden"])
        enc = BoardEncoder(d_model=d, n_layers=int(cfg["n_layers"]), nhead=int(cfg["nhead"]),
                           dim_feedforward=int(cfg["dim_feedforward"]), dropout=float(cfg["dropout"]))
        bands = list(cfg.get("bands", BANDS))
        heads = [BandHead(d, h) for _ in bands]
        value = WDLHead(d_model=d, hidden=h, elo_dim=int(cfg.get("elo_dim", 0)),
                        n_elo_buckets=int(cfg.get("n_elo_buckets", 0)))
        return cls(enc, heads, value, bands=bands)

    def encode(self, packed_pre):
        board = packed_to_board_tensor(packed_pre).to(next(self.parameters()).device)
        return self.encoder(board)

    @staticmethod
    def head_index(elo: torch.Tensor) -> torch.Tensor:
        return ((elo.clamp(1000, 1999) - 1000) // 100).long()
```

- [ ] **Step 4: Run to verify pass**; **Step 5: Commit**
```bash
git add style_policy/multiband_policy.py tests/style_policy/test_multiband_policy.py
git commit -m "feat(multiband): MultiBandPolicy (shared encoder + 10 per-band heads + value head)"
```

---

### Task 2: `train_multiband` (routed training + exports)

**Files:** Create `style_policy/multiband_train.py`; Test add to `tests/style_policy/test_multiband_policy.py`

**Interfaces:**
- Consumes: `MultiBandPolicy`, `PackedMoveDataset`, `masked_square_ce`/`wdl_ce`, `u64_to_mask`, `elo_to_bucket`, `BasePolicy`/`BandHead` (for the export contract).
- Produces: `train_multiband(spec, device, *, resume=False) -> dict`. `spec` is a `load_spec`-style dict with a single stage. Writes under `spec["checkpoint_dir"]`: `{name}.pt` (joint), `{name}_encoder.pt` (BasePolicy-loadable), and `band_heads/{name}_band_{B}.pt` per band.

- [ ] **Step 1: Write the failing smoke test**

Append to `tests/style_policy/test_multiband_policy.py`:
```python
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
```

- [ ] **Step 2: Run to verify it fails** (no module `multiband_train`).

- [ ] **Step 3: Implement** `style_policy/multiband_train.py`:
```python
"""Joint training: shared elo-agnostic encoder + N per-band heads (routed CE) + shared value head.
Mirrors train_one_stage's mechanics (compile/fused-AdamW/cosine/val/resume) with per-band routing."""
from __future__ import annotations
import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from style_policy.multiband_policy import MultiBandPolicy
from style_policy.dataset import PackedMoveDataset
from style_policy.loss import masked_square_ce, wdl_ce
from style_policy.legal_mask import u64_to_mask
from style_policy.model_spec import elo_to_bucket


def _routed_policy_loss(model, squares, hidx, from_sq, to_sq, fmask, tmask, ls):
    B = squares.shape[0]
    fl = squares.new_zeros(()); tl = squares.new_zeros(())
    for g in range(model.n_bands):
        m = hidx == g
        k = int(m.sum())
        if k == 0:
            continue
        sq = squares[m]
        fl = fl + masked_square_ce(model.heads[g].from_logits(sq), from_sq[m], fmask[m], label_smoothing=ls) * k
        tl = tl + masked_square_ce(model.heads[g].to_logits(sq, from_sq[m]), to_sq[m], tmask[m], label_smoothing=ls) * k
    return fl / B, tl / B


def _step(model, batch, device, n_elo, ls, vlw):
    packed = batch["packed_pre"].to(device)
    elo = batch["elo_to_move"]
    hidx = model.head_index(elo).to(device)
    from_sq = batch["from_sq"].to(device); to_sq = batch["to_sq"].to(device)
    fmask = u64_to_mask(batch["from_legal_u64"].to(device))
    tmask = u64_to_mask(batch["to_legal_u64"].to(device))
    result = batch["result"].to(device)
    cls, squares = model.encode(packed)
    fl, tl = _routed_policy_loss(model, squares, hidx, from_sq, to_sq, fmask, tmask, ls)
    vl = wdl_ce(model.value_head(cls, elo_idx=elo_to_bucket(elo, n_elo).to(device)), result)
    return fl + tl + vlw * vl, {"from_ce": float(fl), "to_ce": float(tl), "wdl_ce": float(vl)}


@torch.no_grad()
def _validate(model, val_dl, device, n_elo, use_amp, amp_dtype, vlw):
    was = model.training; model.eval(); tot = 0.0; nb = 0
    for batch in val_dl:
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp and device == "cuda"):
            loss, m = _step(model, batch, device, n_elo, 0.0, vlw)
        tot += m["from_ce"] + m["to_ce"]; nb += 1
    if was:
        model.train()
    return tot / max(nb, 1)


def _export(model, arch, ckpt_dir, name, do_compile):
    sd = model.state_dict()
    if do_compile:
        sd = {k.replace("encoder._orig_mod.", "encoder.", 1): v for k, v in sd.items()}
    enc_sd = {k: v for k, v in sd.items() if k.startswith("encoder.") or k.startswith("value_head.")}
    enc_path = ckpt_dir / f"{name}_encoder.pt"
    torch.save({"architecture": arch, "model": enc_sd}, enc_path)
    torch.save({"architecture": arch, "bands": model.bands, "model": sd}, ckpt_dir / f"{name}.pt")
    d = int(arch["d_model"]); h = int(arch["head_hidden"])
    hd = ckpt_dir / "band_heads"; hd.mkdir(parents=True, exist_ok=True)
    for i, b in enumerate(model.bands):
        pre = f"heads.{i}."
        hsd = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
        torch.save({"band_head": hsd, "d_model": d, "hidden": h,
                    "source_checkpoint": str(enc_path), "band": int(b)}, hd / f"{name}_band_{b}.pt")


def train_multiband(spec: dict, device: str, *, resume: bool = False) -> dict:
    stage = spec["stages"][0]; arch = spec["architecture"]; n_elo = int(arch["n_elo_buckets"])
    name = spec["name"]; ckpt_dir = Path(spec["checkpoint_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    use_amp = bool(stage["use_amp"]); amp_dtype = torch.bfloat16
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    model = MultiBandPolicy.from_config(arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=stage["train"]["learning_rate"],
                            weight_decay=stage["weight_decay"], fused=(device == "cuda"))
    ds = PackedMoveDataset(spec["train_h5"], sample_n=stage["sample"]["n"], seed=stage["sample"]["seed"])
    dl = DataLoader(ds, batch_size=stage["batch_size"], shuffle=True,
                    num_workers=stage["dataloader_num_workers"], collate_fn=PackedMoveDataset.collate)
    val_dl = None
    if spec.get("val_h5") and spec.get("val_sample"):
        vds = PackedMoveDataset(spec["val_h5"], sample_n=spec["val_sample"]["n"], seed=spec["val_sample"]["seed"])
        val_dl = DataLoader(vds, batch_size=stage["batch_size"], shuffle=False,
                            num_workers=stage["dataloader_num_workers"], collate_fn=PackedMoveDataset.collate)
    total_steps = math.ceil(len(ds) / stage["batch_size"]) * int(stage["train"]["epochs"])
    warmup = int(stage.get("warmup_steps", 0)); lr_min = float(stage.get("lr_min_frac", 0.0))
    sched = None
    if str(stage.get("lr_schedule", "constant")) == "cosine":
        def _lam(s):
            if warmup > 0 and s < warmup:
                return (s + 1) / warmup
            p = min(1.0, (s - warmup) / max(1, total_steps - warmup))
            return lr_min + (1.0 - lr_min) * 0.5 * (1.0 + math.cos(math.pi * p))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lam)

    do_compile = bool(stage.get("compile", True)) and device == "cuda"
    if do_compile:
        model.encoder = torch.compile(model.encoder)

    resume_path = ckpt_dir / f"{name}.resume.pt"; step = 0; best = float("inf")
    if resume and resume_path.exists():
        st = torch.load(resume_path, map_location=device)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["optimizer"]); step = int(st["step"]); best = float(st["best"])
        if sched is not None and st.get("scheduler"):
            sched.load_state_dict(st["scheduler"])

    ls = stage.get("label_smoothing", 0.0); vlw = stage.get("value_loss_weight", 1.0)
    val_interval = int(stage.get("val_interval", 0)); ckpt_interval = int(stage.get("checkpoint_interval", 0))
    print(f"multiband train: {name} steps={total_steps} compile={'on' if do_compile else 'off'}")
    model.train(); model.encoder  # noqa
    while step < total_steps:
        for batch in dl:
            if step >= total_steps:
                break
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp and device == "cuda"):
                loss, m = _step(model, batch, device, n_elo, ls, vlw)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), stage["max_gradient_norm"])
            opt.step()
            if sched is not None:
                sched.step()
            if step % stage["log_interval"] == 0:
                print(f"step={step}/{total_steps} from_ce={m['from_ce']:.3f} to_ce={m['to_ce']:.3f} wdl={m['wdl_ce']:.3f}", flush=True)
            if val_dl is not None and val_interval and step > 0 and step % val_interval == 0:
                v = _validate(model, val_dl, device, n_elo, use_amp, amp_dtype, vlw)
                print(f"  [val step={step}] from+to_ce={v:.4f}", flush=True)
                best = min(best, v)
            if ckpt_interval and step > 0 and step % ckpt_interval == 0:
                torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
                            "best": best, "scheduler": sched.state_dict() if sched else None}, resume_path)
            step += 1
    final_val = _validate(model, val_dl, device, n_elo, use_amp, amp_dtype, vlw) if val_dl else float("nan")
    _export(model, arch, ckpt_dir, name, do_compile)
    print(f"saved {ckpt_dir}/{name}.pt + encoder + per-band heads; final_val={final_val:.4f}")
    return {"steps": step, "final_val": final_val}
```

- [ ] **Step 4: Run the smoke test to verify pass** (`pytest tests/style_policy/test_multiband_policy.py -q` → all pass).
- [ ] **Step 5: Commit**
```bash
git add style_policy/multiband_train.py tests/style_policy/test_multiband_policy.py
git commit -m "feat(multiband): routed train_multiband + per-band/encoder exports"
```

---

### Task 3: config + CLI

**Files:** Create `style_policy/model_configs/multiband_16M.yaml`, `scripts/train_multiband.py`

- [ ] **Step 1: Config** `style_policy/model_configs/multiband_16M.yaml` (mirrors `wdl_16M.yaml`):
```yaml
name: multiband_16M
checkpoint_dir: style_policy_checkpoints/multiband_16M
train_h5: /mnt/eloquence_bulk/databases/wdl_training_16M.h5
val_h5: /mnt/eloquence_bulk/databases/wdl_validation_1M.h5
val_sample: {n: 10000, seed: 42}
architecture:
  d_model: 256
  n_layers: 8
  nhead: 8
  dim_feedforward: 1024
  dropout: 0.0
  head_hidden: 512
  elo_dim: 32
  n_elo_buckets: 40
defaults:
  batch_size: 256
  dataloader_num_workers: 4
  use_amp: true
  amp_dtype: bf16
  weight_decay: 0.01
  max_gradient_norm: 1.0
  log_interval: 50
  val_interval: 1000
  checkpoint_interval: 1000
  lr_schedule: cosine
  warmup_steps: 1000
  lr_min_frac: 0.0
  seed: 0
  value_loss_weight: 1.0
stages:
  - sample: {n: 16000000, seed: 1}
    train: {epochs: 1, learning_rate: 0.0002}
    label_smoothing: 0.1
```

- [ ] **Step 2: CLI** `scripts/train_multiband.py`:
```python
#!/usr/bin/env python3
"""Train a MultiBandPolicy (shared encoder + per-band heads)."""
import argparse, torch
from style_policy.model_spec import load_spec
from style_policy.multiband_train import train_multiband

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="multiband_16M")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    rec = train_multiband(load_spec(a.model), a.device, resume=a.resume)
    print(rec)

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Commit**
```bash
git add style_policy/model_configs/multiband_16M.yaml scripts/train_multiband.py
git commit -m "feat(multiband): multiband_16M config + train CLI"
```

---

### Task 4: Run + eval (GPU, gated)

**Files:** none (execution + interpretation). Precondition: GPU free.

- [ ] **Step 1: Train** `PYTHONPATH=. python -m scripts.train_multiband --model multiband_16M --device cuda` (~3h). Confirm it saves `multiband_16M.pt`, `multiband_16M_encoder.pt`, and `band_heads/multiband_16M_band_{1000..1900}.pt`.
- [ ] **Step 2: Strength ladder** — `PYTHONPATH=. python -m scripts.rate_band_heads --checkpoint style_policy_checkpoints/multiband_16M/multiband_16M_encoder.pt --head-dir style_policy_checkpoints/multiband_16M/band_heads --prefix multiband_16M_band_ --bands 1000 1500 1900 --temperature 0.1 --games-per-level 30 --device cuda` (and `--temperature 1.0`).
- [ ] **Step 3: Compare + record** — vs the frozen ladder (1567/1780/1828) and `wdl_16M`'s flat conditioning. Win = steeper/monotonic ladder, high end > ~1800. Optionally per-band move-match via `eval_band_head_row` per exported head. Write the verdict to memory `diagonal-findings.md`.

---

## Self-Review
- **Coverage:** model (T1), routed training + exports (T2), config + CLI (T3), run + eval (T4).
- **Type consistency:** `_export` writes `{band_head,d_model,hidden,source_checkpoint,band}` (consumed by `BandHead`/`BandHeadBot`/`rate_band_heads`) and a `{architecture,model}` encoder ckpt (consumed by `BasePolicy.load_state_dict(strict=False)`); head-prefix `heads.{i}.` stripped to bare `from_head.`/`to_head.` keys matching `BandHead`.
- **Reuse:** encoder/heads/value/loss/masking/dataset all reused; routing is the only new logic.
- **compile safety:** only the encoder is compiled; routing runs eager; `_orig_mod.` stripped on export.
