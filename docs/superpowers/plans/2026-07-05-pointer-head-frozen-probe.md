# Query/Key Pointer Head — Frozen-Encoder Probe Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or executing-plans to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Get an early, cheap signal on whether a query/key pointer policy head beats the current factored MLP head, by training fresh heads on a *frozen* trained encoder and comparing joint top-1 move-match on 2025-05 — including a CLS on/off A/B.

**Architecture:** Freeze a trained `MultiBandPolicy` encoder. On its frozen square-token features, train three fresh **shared** (all-bands pooled, elo-agnostic) heads: (1) factored FromHead/ToHead baseline, (2) pointer + CLS, (3) pointer no-CLS. Evaluate all three with joint decoding on the 2025-05 val set and compare. Reuses the existing frozen-head machinery in `style_policy/band_head.py`.

**Tech Stack:** PyTorch, python-chess, h5py; existing `MultiBandPolicy`, `PackedMoveDataset`, `packed_to_board`, `legal_from_u64/legal_to_u64`.

## Global Constraints
- **Elo-agnostic encoder:** the probe never feeds elo to the encoder; heads are shared (no band split needed for the probe — per-band ≈ shared for prediction, and pooling gives each head more data). Verbatim: no elo/opp-elo into `encode()`.
- **Fair baseline:** compare the pointer against a factored head *retrained on the same frozen encoder*, NOT the checkpoint's co-adapted heads. (Optionally also report the co-adapted head as a conservative floor.)
- **Frozen-probe reading:** it is biased *against* the pointer (encoder co-adapted to factored + no encoder joint-adaptation). Tie/win = green light; small loss = inconclusive; big loss = likely worse.
- **Promotion piece is NOT modeled** in the pointer for the probe — move-match is (from,to) only (matches `eval_band_head_row`, which teacher-checks from & to).
- **Metric:** joint top-1 move-match on `/mnt/eloquence_bulk/databases/wdl_validation_2025_05.h5`.
- **History:** the big/GAB encoders were trained with history (`n_ply=2`); the probe MUST pass history to `encode()` for realistic features. (Note: the existing `train_band_head`/`eval_band_head_row` call `encode(packed)` WITHOUT history — do not copy that omission.)
- **GPU:** the box is running the big+GAB job (~3 days). Implement + unit-test now (CPU, tiny models). Run the actual probe after the GPU frees (or carefully alongside). Prefer the **GAB-big encoder** when done; use the existing **big** encoder or the **16M** encoder for an earlier look.
- Container tests: `docker exec -e PYTHONPATH=. -w /workspaces/eloquent-encoding 1ec2b8ce64c8 python -m pytest`. No `tests/**/__init__.py`.

---

## File Structure
- **Create** `style_policy/pointer_head.py` — `PointerHead` module + `joint_ce` loss.
- **Modify** `style_policy/board_encode.py` — add `legal_move_matrix(board) -> np.ndarray (64,64) bool`.
- **Create** `scripts/probe_pointer_head.py` — CLI harness: freeze encoder, train the 3 head variants, eval joint move-match, print the comparison.
- **Create** `tests/style_policy/test_pointer_head.py` — unit tests for the module + helper.

---

### Task 1: `legal_move_matrix` helper

**Files:**
- Modify: `style_policy/board_encode.py` (add function near `legal_from_u64`, line ~61)
- Test: `tests/style_policy/test_pointer_head.py`

**Interfaces:**
- Produces: `legal_move_matrix(board: chess.Board) -> np.ndarray` — bool (64,64), `[from,to]=True` iff some legal move goes from→to (promotions collapse to their shared from/to).

- [ ] **Step 1: Write the failing test**
```python
import numpy as np, chess
from style_policy.board_encode import legal_move_matrix, legal_from_u64

def test_legal_move_matrix_matches_board():
    board = chess.Board()  # startpos: 20 legal moves, 16 distinct (from,to)
    m = legal_move_matrix(board)
    pairs = {(mv.from_square, mv.to_square) for mv in board.legal_moves}
    assert m.sum() == len(pairs)
    for (fr, to) in pairs:
        assert m[fr, to]
    # rows with any legal to must equal the legal-from bitboard
    froms = {i for i in range(64) if m[i].any()}
    lf = legal_from_u64(board)
    assert froms == {i for i in range(64) if (lf >> i) & 1}
```

- [ ] **Step 2: Run it, verify it fails** (`ImportError: legal_move_matrix`)

- [ ] **Step 3: Implement**
```python
def legal_move_matrix(board: "chess.Board") -> np.ndarray:
    """(64,64) bool: [from,to]=True iff some legal move goes from->to (promo piece ignored)."""
    m = np.zeros((64, 64), dtype=bool)
    for mv in board.legal_moves:
        m[mv.from_square, mv.to_square] = True
    return m
```

- [ ] **Step 4: Run test, verify PASS**
- [ ] **Step 5: Commit** (`feat(board_encode): legal_move_matrix for joint policy masking`)

---

### Task 2: `PointerHead` module + `joint_ce`

**Files:**
- Create: `style_policy/pointer_head.py`
- Test: `tests/style_policy/test_pointer_head.py`

**Interfaces:**
- Produces:
  - `PointerHead(d_model:int, d_head:int=64, n_heads:int=1, use_cls:bool=True)`; `forward(squares (B,64,d), cls (B,d)|None) -> logits (B,64,64)` where `logits[b,i,j]` = score for move i→j.
  - `joint_ce(logits (B,64,64), from_sq (B,), to_sq (B,), legal_mat (B,64,64) bool, label_smoothing=0.0) -> scalar`.

- [ ] **Step 1: Write failing tests**
```python
import torch
from style_policy.pointer_head import PointerHead, joint_ce

def test_pointer_shapes_and_cls():
    h = PointerHead(d_model=32, d_head=16, n_heads=2, use_cls=True)
    sq = torch.randn(4, 64, 32); cls = torch.randn(4, 32)
    assert h(sq, cls).shape == (4, 64, 64)
    try:
        h(sq, None); assert False
    except ValueError:
        pass

def test_pointer_no_cls_ignores_cls():
    h = PointerHead(d_model=32, use_cls=False)
    sq = torch.randn(2, 64, 32)
    assert torch.allclose(h(sq, None), h(sq, torch.randn(2, 32)))

def test_joint_ce_masks_and_runs():
    h = PointerHead(d_model=32, use_cls=False)
    sq = torch.randn(2, 64, 32)
    logits = h(sq, None)
    legal = torch.zeros(2, 64, 64, dtype=torch.bool)
    legal[0, 8, 16] = True; legal[0, 8, 24] = True; legal[1, 1, 18] = True
    fr = torch.tensor([8, 1]); to = torch.tensor([16, 18])
    loss = joint_ce(logits, fr, to, legal)
    assert torch.isfinite(loss)
    # argmax over legal is always a legal (from,to)
    flat = logits.masked_fill(~legal, float("-inf")).view(2, -1)
    idx = flat.argmax(-1)
    assert bool(legal[0, idx[0] // 64, idx[0] % 64]) and bool(legal[1, idx[1] // 64, idx[1] % 64])
```

- [ ] **Step 2: Run, verify fail**

- [ ] **Step 3: Implement `style_policy/pointer_head.py`**
```python
"""Query/key pointer policy head: joint P(from,to) via sum_h q_i^h . k_j^h over 64 square tokens.
Contrast with the factored FromHead/ToHead. Promotion piece is NOT modeled (probe = from/to only).
Optional CLS injected as a global conditioner added to every square token before the projections."""
from __future__ import annotations
import torch
import torch.nn as nn

class PointerHead(nn.Module):
    def __init__(self, d_model: int, d_head: int = 64, n_heads: int = 1, use_cls: bool = True):
        super().__init__()
        self.n_heads = int(n_heads); self.d_head = int(d_head); self.use_cls = bool(use_cls)
        self.q_proj = nn.Linear(d_model, n_heads * d_head)
        self.k_proj = nn.Linear(d_model, n_heads * d_head)
        if use_cls:
            self.cls_proj = nn.Linear(d_model, d_model)
        self.scale = d_head ** -0.5

    def forward(self, squares: torch.Tensor, cls: torch.Tensor | None = None) -> torch.Tensor:
        b = squares.shape[0]
        feats = squares
        if self.use_cls:
            if cls is None:
                raise ValueError("PointerHead(use_cls=True) requires cls")
            feats = squares + self.cls_proj(cls).unsqueeze(1)   # global conditioner, broadcast
        q = self.q_proj(feats).view(b, 64, self.n_heads, self.d_head)
        k = self.k_proj(feats).view(b, 64, self.n_heads, self.d_head)
        return torch.einsum("bihd,bjhd->bij", q, k) * self.scale   # (B,64,64)

def joint_ce(logits, from_sq, to_sq, legal_mat, label_smoothing: float = 0.0):
    b = logits.shape[0]
    flat = logits.masked_fill(~legal_mat, float("-inf")).view(b, -1)
    target = (from_sq.long() * 64 + to_sq.long())
    return torch.nn.functional.cross_entropy(flat, target, label_smoothing=label_smoothing)
```

- [ ] **Step 4: Run tests, verify PASS**
- [ ] **Step 5: Commit** (`feat(pointer_head): query/key joint policy head + joint_ce`)

---

### Task 3: Probe harness `scripts/probe_pointer_head.py`

**Files:**
- Create: `scripts/probe_pointer_head.py`
- (No unit test — this is integration glue; validated by the Task 4 smoke run.)

**Interfaces (CLI):**
`--ckpt <MultiBandPolicy .pt>  --variant {factored,pointer_cls,pointer_nocls}  --train-h5 ... --val-h5ault 2025_05  --sample-n <int>  --steps <int>  --batch 256  --eval-n 20000  --out <head.pt>`

**Key implementation points (mirror `band_head.train_band_head` / `eval_band_head_row`, with these differences):**
- Load **`MultiBandPolicy`** (not BasePolicy): `m = MultiBandPolicy.from_config(ck["architecture"]); m.load_state_dict(ck["model"]); m.eval(); freeze all params`.
- **Pass history to the encoder.** Slice dataset `hist_from/to/cap` to `arch["n_history_ply"]`; `cls, squares = m.encode(packed, hist=(hf,ht,hc))` (skip hist only if `not arch.get("use_last_move")`).
- **Single shared head** trained on all-bands data (`PackedMoveDataset(train_h5, sample_n=..., seed=1)` with no `band=` filter).
- **Variant wiring:**
  - `factored`: fresh `BandHead(d, head_hidden)`; loss = `masked_square_ce(from_logits, from_sq, fmask) + masked_square_ce(to_logits, to_sq, tmask)` (reuse existing masks from the dataset).
  - `pointer_cls` / `pointer_nocls`: `PointerHead(d, d_head=64, n_heads=1, use_cls=<bool>)`; build per-batch `legal_mat (B,64,64)` via `legal_move_matrix(packed_to_board(packed_i))` (do this in a small collate/wrapper so it runs in dataloader workers); loss = `joint_ce(logits, from_sq, to_sq, legal_mat)`.
- Training loop otherwise identical to `train_band_head` (AdamW, bf16 autocast on cuda, `encode` under `no_grad`, head under grad). Default `steps` sized to ~1 epoch over `sample-n` (e.g. sample-n=8_000_000).
- Save head + meta (`variant`, `source_checkpoint`, `d_model`, `d_head`, `n_heads`, `use_cls`).

- [ ] **Step 1:** Write the loader + freeze + dataset/history plumbing.
- [ ] **Step 2:** Implement the per-batch `legal_mat` builder (wrapper dataset or collate calling `legal_move_matrix`).
- [ ] **Step 3:** Implement the three variant train branches (factored / pointer±cls).
- [ ] **Step 4:** Add `--out` save. Commit (`feat(probe): frozen-encoder pointer-vs-factored trainer`).

---

### Task 4: Joint move-match eval + comparison, and smoke run

**Files:**
- Modify: `scripts/probe_pointer_head.py` (add `evaluate()` + a `--compare` mode that trains/loads all 3 and prints the table)

**Interfaces:**
- `evaluate(model, head, variant, val_h5, n) -> float` (joint top-1 move-match %).

**Key points (mirror `eval_band_head_row`, add joint decode + history):**
- Per val position: `board = packed_to_board(packed_i)`; skip `board.is_game_over()`.
- `cls, squares = model.encode(pk, hist=...)`.
- **factored:** joint-decode = enumerate legal moves, `P(from)*P(to|from)` via the head's `from_logits`/`to_logits`, argmax over legal moves (reuse the logic in `multiband_bot.MultiBandBot` joint path).
- **pointer:** `logits (1,64,64)`; `legal = legal_move_matrix(board)`; `pred = logits.masked_fill(~legal, -inf).view(-1).argmax()`; `pf, pt = pred//64, pred%64`.
- match iff `pf==from_sq[i] and pt==to_sq[i]`.

- [ ] **Step 1:** Write `evaluate()` for both head types (joint).
- [ ] **Step 2:** `--compare` mode: run factored / pointer_cls / pointer_nocls, print:
  ```
  variant        train_steps  move%_2025_05
  factored(fresh)   ...          XX.XX
  pointer + CLS     ...          XX.XX
  pointer  no-CLS   ...          XX.XX
  (reference: co-adapted factored head from ckpt = YY.YY ; full model = ZZ.ZZ)
  ```
- [ ] **Step 3: Smoke run** on the **16M** encoder, tiny (`--sample-n 200000 --eval-n 3000`) to validate end-to-end on CPU/GPU without waiting for the big job. Expected: all three produce finite, non-degenerate move% (>0, pointer not at chance). Fix any wiring bugs.
- [ ] **Step 4: Commit** (`feat(probe): joint move-match eval + 3-way comparison`).

---

## Interpretation (decision rule)
- **pointer (either CLS variant) ≥ fresh-factored** → green light: run a full from-scratch co-trained pointer model (expect ≥ this, since the frozen probe handicaps the pointer).
- **pointer clearly < fresh-factored** → the readout is likely worse; drop or reconsider (try multi-head pointer `n_heads>1` first — cheap).
- **pointer_cls vs pointer_nocls** → the CLS value you never measured. Expect `cls ≥ nocls` on the frozen encoder (side-to-move likely parked in CLS). If ~equal even here → CLS is droppable in the co-trained version.

## Real run (after GPU frees)
`--ckpt` = GAB-big (preferred) or big; `--sample-n 8_000_000` (≈1 epoch), `--eval-n 20000`, run all 3 variants via `--compare`.

## Self-review notes
- Types consistent: `legal_mat` bool (B,64,64); `joint_ce` target = from*64+to; eval decodes pred//64, pred%64. ✔
- Elo-agnostic preserved (shared head, no elo to encoder). ✔
- History passed to `encode` (unlike the copied `band_head` functions). ✔
- Promotion excluded consistently in train + eval (from/to only). ✔
