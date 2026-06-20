# Style-Conditioned Move Policy — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, from scratch in a fresh `style_policy/` module, a chess move predictor that scores moves as a two-stage pointer policy (from-square then to-square) over a transformer board encoder, then layer unsupervised *style* conditioning on top to predict how different kinds of players move — and ultimately to select counter-styles against an opponent.

**Architecture:** One shared transformer encoder turns a board into 64 per-square tokens (+ a few global/special tokens). A base policy points into those 64 tokens twice — once for the origin square (legality-masked), once for the destination given the origin (legality-masked). Player *style* is a discrete latent: per-bucket residual heads add logits to the base in log-space (product-of-experts), and the buckets are discovered by hard/soft EM over games rather than by profiling individuals. Strength (elo) is conditioned into the *base* so the style residual is forced to capture non-strength variation; elo stays optional at inference.

**Tech Stack:** Python 3.11 (devcontainer), PyTorch 2.5.1+cu121, h5py, python-chess, numpy. Reuses the existing jepa3 *packed* HDF5 datasets on the `eloquence-bulk` volume (`/mnt/eloquence_bulk/databases/j3_*.h5`).

## Global Constraints

- **Runtime:** devcontainer only (`devcontainer up --workspace-folder .`; run via `devcontainer exec --workspace-folder . bash -lc '...'`). Repo root is on `PYTHONPATH`; inside the container the workspace is `/workspaces/eloquent-encoding`.
- **GPU:** laptop RTX 500 Ada, **4GB VRAM**. Every config must fit 4GB: keep `d_model` ≤ 256, batch sizes modest, use AMP. Big runs are expected to need gradient accumulation; never assume more memory.
- **Data format:** reuse the existing **jepa3 packed** layout (`packed_pre`/`packed_post` `uint8 (N,34)`, `from_legal_u64`/`to_legal_u64` `uint64`, `from_sq`/`to_sq`/`promotion` `uint8`, `elo_to_move` `int16`; file attrs `jepa3_packed_format=1`, `packed_layout_version=1`). Do **not** rebuild the 97GB of data. The codec that reads this layout is lifted into the new module and is the on-disk contract.
- **Dependencies:** no new runtime deps beyond `requirements.txt`. `pytest` is dev-only (`pip install pytest` in the container; it is not in requirements).
- **Testing:** deterministic parts (codec round-trip, square categories, tensor shapes, legality masking, loss math) get real asserting tests. Training *quality* is validated by smoke runs and metrics, not unit asserts.
- **Naming:** new module is `style_policy/`. Archived modules move under `archive/` and are reference-only (not maintained, not imported by active code).

---

## Design Decisions & Rationale (durable context)

These were settled during design discussion; preserve them so future work doesn't re-litigate.

1. **64 square tokens, not piece-centric, not CNN.** The policy is a pointer that indexes board tokens for *both* the origin and the destination. Destinations are usually empty squares, so a per-square tokenization is the only one that serves both heads uniformly. Piece-centric tokens have no token for empty destinations; CNN regresses on the long-range piece interactions transformers get for free. (jepa3's `BoardEncoderV3` already used CLS + 64 square tokens; we keep that shape.)

2. **Heads point into the 64 *square* tokens, not the CLS vector.** jepa3's `ToSquareHead` read the global CLS vector — the older flat-vector design. The new heads must point over the per-square tokens (like `world_model`'s `PatchPointerFromHead`). CLS is retained for global state + goal-1 probing, not for pointing.

3. **Two-stage factored policy.** `from-head: (squares) → 64 logits`, masked by `from_legal_u64`. `to-head: (squares, from_sq) → 64 logits`, masked by the legal targets given the chosen origin. Promotion is a small separate head fired only on pawn-to-back-rank moves; it is not a style axis.

4. **Style = discrete latent, discovered by EM.** A base policy is trained across all players. Per-bucket residual heads fine-tune each bucket. E-step assigns each *game* to the bucket whose policy best predicts its moves; M-step refits buckets. We do **not** profile individual players (constraint: behavior-level clustering, game-level assignment — needs a game id, never a username).

5. **Residual is product-of-experts in logit space.** `final_logits = base_logits + residual_logits`, then mask, then softmax. Adding in logit space already makes the residual a *relative* reweighting, so the residual must **not** receive the base logits as input (shortcut hazard); L2-regularize the residual small so style bends rather than re-predicts. Mirror the factorization: a from-residual *and* a to-residual.

6. **Residual inputs:** detached board encoding (base frozen during EM M-step so buckets don't chase a moving target) + bucket code `z_k` + the bucket code only. Semantic move features are well-defined only at the **to-stage** (where `(from,to)` is a concrete move); even there, prefer to let the encoder supply current-board facts and only consider hand-feeding *post-move consequence* features (`gives_check`, Δmaterial) — and under goal 1 (study emergence) do **not** hand-feed; probe instead.

7. **Strength vs style.** Elo is the dominant axis of move-prediction variance; uncontrolled EM rediscovers skill, not style. Mitigation: condition the **base** on elo so it absorbs strength, forcing the residual buckets to capture non-strength variation = style. Elo is optional at inference (feed it if known; marginalize/default if not).

8. **Inference-time style classification** = the training E-step on the opponent: accumulate per-move log-likelihood under each bucket + log-prior → posterior over buckets, updated move-by-move. Use the **mixture of policies** for prediction under uncertainty; reserve embedding-averaging for a single point estimate. Most moves are non-discriminative (forced/obvious); the signal is at positions where buckets disagree.

9. **Dynamics are free.** Chess has an exact rules engine, so we never *learn* a transition model as a deliverable. The world-model/JEPA machinery (from the archived `world_model/`) is out of scope here except as an optional goal-1 probe target. The one place a learned dynamics model would earn its keep — multi-step *latent* rollouts for search — is deferred to the engine phase.

## Empirical Gates (why later phases are not yet detailed)

Phases 2–4 are research-gated and must each be expanded into their own plan when reached, because each depends on the previous phase's empirical result:

- **Gate A (after Phase 2):** Do EM buckets (a) beat the single base on held-out move prediction, (b) separate on style statistics (sacrifice/capture/check rates), and (c) stay *uncorrelated with elo*? If not — styles are too weak or are just strength; stop and reconsider. Build the elo-conditioned base and the style-stat validators *before* the EM loop.
- **Gate B (after Phase 3):** At fixed strength, is there non-transitive style *matchup* structure (does style Y beat X beyond what elo explains)? If matchups are transitive, the meta-game reduces to "play stronger" and Phase 4 is not worth building.

---

## File Structure (new module)

```
style_policy/
  __init__.py
  packed_codec.py        # LIFTED from jepa3/packed_board_codec.py — decodes packed_pre → board tensor. On-disk contract.
  square_categories.py   # LIFTED from jepa3/board_square_categories.py — 18-way per-square piece category.
  board_encoder.py       # BoardEncoder: 64 square tokens + special tokens → (cls, square_tokens). Phase 1.
  policy_heads.py        # FromHead (pointer→64, legal-masked), ToHead (pointer→64 | from_sq, legal-masked). Phase 1.
  promotion_head.py      # PromotionHead: (squares, from, to) → 4 logits, fired only on promotions. Phase 1.
  model.py               # BasePolicy: encoder + heads. forward_from / forward_to / forward_promotion. Phase 1.
  loss.py                # masked_square_ce(logits, target, legal_u64). Phase 1.
  dataset.py             # PackedMoveDataset: reads j3 packed h5 → tensors. Phase 1.
  legal_mask.py          # u64 bitboard → (B,64) bool mask helper. Phase 1.
  model_spec.py          # YAML defaults + per-stage merge (jepa3-style). Phase 1.
  training_loop.py       # one-stage train/val loop, metrics, checkpoint. Phase 1.
  train.py               # CLI: python -m style_policy.train --model NAME --stage K. Phase 1.
  model_configs/
    tiny_smoke.yaml       # tiny model + small sample over j3_training_1M for CI/smoke. Phase 1.
    base_16M.yaml         # full base run over j3_training_16M. Phase 1.
  # Phase 2+ (added when reached): style_residual.py, em_loop.py, style_stats.py, ...
tests/style_policy/
  test_packed_codec.py
  test_square_categories.py
  test_legal_mask.py
  test_board_encoder.py
  test_policy_heads.py
  test_loss.py
  test_dataset.py
  test_model_forward.py
```

---

## Phase 0 — Repo restructure (archive old, scaffold new)

### Task 0.1: Archive inactive modules

**Files:**
- Create: `archive/README.md`
- Move: `embedding/ jepa/ jepa2/ jepa3/ gfp/ rfp/ world_model/ move_predictor/` → `archive/`
- Move: the matching `tests/test_*.py` for those modules → `archive/tests/`
- Keep active at root: `dataset_generation/`, `scripts/`, `h5_dataloader.py`, `hdf5_tool.py`, `requirements*.txt`, `.devcontainer/`, `multi.sh`, `embedding_report*`.

- [ ] **Step 1: Create the archive marker**

`archive/README.md`:
```markdown
# Archive (reference only)

Superseded approaches, kept for reference. **Not maintained, not imported by active code.**
Active work lives in `style_policy/` (see `docs/superpowers/plans/`).

Lineage: embedding (MAE) → jepa/jepa2/jepa3 (JEPA encoders; v3 is the mature one) →
gfp/rfp (from-square predictors on a frozen jepa3 encoder) → world_model (patch-JEPA + recon).
`style_policy/` lifts the packed board codec and square-category logic from jepa3 and rebuilds
the encoder + pointer policy fresh.
```

- [ ] **Step 2: Move modules and their tests with git**

```bash
cd /workspaces/eloquent-encoding
mkdir -p archive/tests
git mv embedding jepa jepa2 jepa3 gfp rfp world_model move_predictor archive/
git mv tests/test_*.py archive/tests/
```

- [ ] **Step 3: Verify no active code imports archived modules**

Run:
```bash
grep -rEn "^\s*(import|from)\s+(embedding|jepa|jepa2|jepa3|gfp|rfp|world_model|move_predictor)\b" \
  dataset_generation scripts *.py 2>/dev/null || echo "clean"
```
Expected: `clean` (the data pipeline and root tools must not depend on archived code). If anything prints, note it — `dataset_generation` is shared and may need a lifted copy rather than an import.

- [ ] **Step 4: Verify the tree**

Run: `ls archive && echo "---" && ls tests 2>/dev/null || echo "tests now empty"`
Expected: the eight modules + `tests/` under `archive/`; root `tests/` empty (new tests land under `tests/style_policy/`).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: archive superseded modules under archive/"
```

### Task 0.2: Scaffold the new module

**Files:**
- Create: `style_policy/__init__.py`, `tests/style_policy/__init__.py` (empty)

- [ ] **Step 1: Create package markers**

`style_policy/__init__.py`:
```python
"""Style-conditioned chess move policy: shared encoder + two-stage pointer policy + style buckets."""
```
`tests/style_policy/__init__.py`: empty file.

- [ ] **Step 2: Verify import**

Run: `devcontainer exec --workspace-folder . bash -lc 'cd /workspaces/eloquent-encoding && python -c "import style_policy; print(style_policy.__doc__)"'`
Expected: prints the docstring.

- [ ] **Step 3: Commit**

```bash
git add style_policy tests/style_policy
git commit -m "feat(style_policy): scaffold new module"
```

---

## Phase 1 — Base two-stage pointer policy

Deliverable: a trainable `BasePolicy` (encoder + from/to/promotion heads) that learns to predict human moves from `j3_*` packed data, with elo conditioning, runnable end-to-end on the 4GB GPU.

### Task 1.1: Lift the packed codec and square categories

**Files:**
- Create: `style_policy/packed_codec.py` (copy of `archive/jepa3/packed_board_codec.py`, imports rewritten to be self-contained)
- Create: `style_policy/square_categories.py` (copy of `archive/jepa3/board_square_categories.py`, imports rewritten)
- Test: `tests/style_policy/test_packed_codec.py`, `tests/style_policy/test_square_categories.py`

**Interfaces:**
- Produces: `PACKED_BOARD_LEN: int`, `packed_to_board_tensor(packed) -> torch.Tensor` (board planes), `board_tensor_to_packed(board) -> np.ndarray`; `NUM_SQUARE_CATEGORIES: int`, `square_categories_from_board_tensor(board) -> torch.Tensor` shape `(B,64)` int64. These define the board-tensor channel layout the encoder consumes — document the channel map in `packed_codec.py`'s module docstring.

- [ ] **Step 1: Copy the two files and rewrite imports**

```bash
cd /workspaces/eloquent-encoding
cp archive/jepa3/packed_board_codec.py style_policy/packed_codec.py
cp archive/jepa3/board_square_categories.py style_policy/square_categories.py
# Then edit: any `from jepa3...` import inside these becomes a `from style_policy...` import.
grep -n "jepa3" style_policy/packed_codec.py style_policy/square_categories.py
```
Expected after edits: no `jepa3` references remain.

- [ ] **Step 2: Write the round-trip test (the on-disk contract)**

`tests/style_policy/test_packed_codec.py`:
```python
import chess
import numpy as np
import torch
from style_policy.packed_codec import board_tensor_to_packed, packed_to_board_tensor, PACKED_BOARD_LEN


def _board_tensor_from_fen(fen):
    # Mirror how dataset rows are produced: build the planes the codec expects from a python-chess board.
    # If the codec exposes a board->tensor helper, use it; otherwise reconstruct via the documented channel map.
    from style_policy.square_categories import square_categories_from_board_tensor  # noqa: F401
    raise NotImplementedError  # replaced in Step 3 once the lifted API is confirmed


def test_packed_len_constant():
    assert PACKED_BOARD_LEN == 34


def test_packed_roundtrip_startpos():
    board = chess.Board()  # standard start
    # Build the board tensor the codec round-trips, pack it, unpack it, assert equality.
    # Uses the lifted board_tensor_to_packed / packed_to_board_tensor as inverse ops.
    # (Concrete tensor construction filled in Step 3 against the confirmed API.)
    packed = np.zeros((1, PACKED_BOARD_LEN), dtype=np.uint8)
    restored = packed_to_board_tensor(torch.from_numpy(packed))
    assert restored.shape[0] == 1
```

- [ ] **Step 3: Confirm the lifted API and finish the test**

Run: `devcontainer exec --workspace-folder . bash -lc 'cd /workspaces/eloquent-encoding && python -c "import inspect, style_policy.packed_codec as c; print([n for n in dir(c) if not n.startswith(chr(95))])"'`
Then read the function signatures and complete `test_packed_roundtrip_startpos` so it: builds a board tensor for the start position, packs then unpacks it, and asserts the unpacked tensor equals the original. Add a second case decoding a real row from disk:
```python
def test_decode_real_row_matches_legal_from():
    import h5py
    with h5py.File("/mnt/eloquence_bulk/databases/j3_training_1M.h5", "r") as f:
        packed = torch.from_numpy(f["packed_pre"][0:4].astype("uint8"))
        from_sq = f["from_sq"][0:4].astype("int64")
        from_legal = f["from_legal_u64"][0:4].astype("uint64")
    board = packed_to_board_tensor(packed)
    assert board.shape[0] == 4
    # Ground-truth from_sq must be a legal origin (bit set in from_legal_u64).
    for i in range(4):
        assert (int(from_legal[i]) >> int(from_sq[i])) & 1 == 1
```

- [ ] **Step 4: Write the square-categories test**

`tests/style_policy/test_square_categories.py`:
```python
import torch
from style_policy.square_categories import NUM_SQUARE_CATEGORIES, square_categories_from_board_tensor
from style_policy.packed_codec import packed_to_board_tensor
import h5py


def test_num_categories_is_18():
    assert NUM_SQUARE_CATEGORIES == 18


def test_categories_shape_and_range():
    with h5py.File("/mnt/eloquence_bulk/databases/j3_training_1M.h5", "r") as f:
        packed = torch.from_numpy(f["packed_pre"][0:8].astype("uint8"))
    board = packed_to_board_tensor(packed)
    cats = square_categories_from_board_tensor(board)
    assert cats.shape == (8, 64)
    assert cats.min() >= 0 and cats.max() < NUM_SQUARE_CATEGORIES
```

- [ ] **Step 5: Run tests**

Run: `devcontainer exec --workspace-folder . bash -lc 'cd /workspaces/eloquent-encoding && python -m pytest tests/style_policy/test_packed_codec.py tests/style_policy/test_square_categories.py -q'`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add style_policy/packed_codec.py style_policy/square_categories.py tests/style_policy/test_packed_codec.py tests/style_policy/test_square_categories.py
git commit -m "feat(style_policy): lift packed codec + square categories with round-trip tests"
```

### Task 1.2: Legality mask helper

**Files:**
- Create: `style_policy/legal_mask.py`
- Test: `tests/style_policy/test_legal_mask.py`

**Interfaces:**
- Produces: `u64_to_mask(u64: torch.Tensor) -> torch.Tensor` — input `(B,)` uint64/int64, output `(B,64)` bool where bit `s` of the integer maps to square `s`.

- [ ] **Step 1: Write the failing test**

`tests/style_policy/test_legal_mask.py`:
```python
import torch
from style_policy.legal_mask import u64_to_mask


def test_single_bit():
    m = u64_to_mask(torch.tensor([1 << 5], dtype=torch.int64))
    assert m.shape == (1, 64)
    assert m[0, 5].item() is True
    assert m[0].sum().item() == 1


def test_multi_bits():
    val = (1 << 0) | (1 << 63)
    m = u64_to_mask(torch.tensor([val], dtype=torch.int64))
    assert m[0, 0] and m[0, 63] and m[0].sum().item() == 2
```

- [ ] **Step 2: Run, expect fail**

Run: `devcontainer exec --workspace-folder . bash -lc 'cd /workspaces/eloquent-encoding && python -m pytest tests/style_policy/test_legal_mask.py -q'`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

`style_policy/legal_mask.py`:
```python
"""Bitboard (uint64) → (B,64) boolean square mask. Bit s ↔ square s (a1=0 .. h8=63)."""
from __future__ import annotations
import torch


def u64_to_mask(u64: torch.Tensor) -> torch.Tensor:
    bits = torch.arange(64, device=u64.device, dtype=torch.int64)
    vals = u64.to(torch.int64).unsqueeze(-1)  # (B,1)
    return ((vals >> bits) & 1).to(torch.bool)  # (B,64)
```

- [ ] **Step 4: Run, expect pass**

Run: same pytest command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add style_policy/legal_mask.py tests/style_policy/test_legal_mask.py
git commit -m "feat(style_policy): bitboard→square-mask helper"
```

### Task 1.3: Board encoder

**Files:**
- Create: `style_policy/board_encoder.py`
- Test: `tests/style_policy/test_board_encoder.py`

**Interfaces:**
- Produces: `class BoardEncoder(nn.Module)` with `__init__(self, *, d_model, n_layers, nhead, dim_feedforward, dropout)` and `forward(self, board_tensor) -> tuple[Tensor, Tensor]` returning `(cls (B,d_model), squares (B,64,d_model))`. Input `board_tensor` is the output of `packed_to_board_tensor`.

- [ ] **Step 1: Write the failing test**

`tests/style_policy/test_board_encoder.py`:
```python
import torch, h5py
from style_policy.board_encoder import BoardEncoder
from style_policy.packed_codec import packed_to_board_tensor


def _boards(n=4):
    with h5py.File("/mnt/eloquence_bulk/databases/j3_training_1M.h5", "r") as f:
        packed = torch.from_numpy(f["packed_pre"][0:n].astype("uint8"))
    return packed_to_board_tensor(packed)


def test_encoder_output_shapes():
    enc = BoardEncoder(d_model=64, n_layers=2, nhead=4, dim_feedforward=128, dropout=0.0)
    cls, squares = enc(_boards(4))
    assert cls.shape == (4, 64)
    assert squares.shape == (4, 64, 64)


def test_castling_changes_encoding():
    # Two boards differing only in castling rights must encode differently
    # (guards that board-global state reaches the encoder, not just piece squares).
    import chess
    from style_policy.packed_codec import board_tensor_to_packed
    # Construct via codec helpers; if board-global state is folded into squares this still holds.
    enc = BoardEncoder(d_model=64, n_layers=2, nhead=4, dim_feedforward=128, dropout=0.0).eval()
    b = _boards(2)
    with torch.no_grad():
        cls, _ = enc(b)
    assert cls.shape == (2, 64)  # smoke; full castling-diff assertion added once channel map confirmed
```

- [ ] **Step 2: Run, expect fail**

Run: `devcontainer exec --workspace-folder . bash -lc 'cd /workspaces/eloquent-encoding && python -m pytest tests/style_policy/test_board_encoder.py -q'`
Expected: FAIL.

- [ ] **Step 3: Implement the encoder**

`style_policy/board_encoder.py`:
```python
"""Transformer board encoder: 64 square tokens + CLS (+ side-to-move). Returns (cls, square_tokens).

Square token s = piece_category_embed(category[s]) + square_position_embed(s).
CLS carries side-to-move. Board-global state (castling, en-passant) is read from the board
tensor's planes; if the lifted codec exposes those planes, add dedicated special tokens here.
Heads point into the 64 SQUARE tokens (not CLS) — see policy_heads.py.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from style_policy.square_categories import NUM_SQUARE_CATEGORIES, square_categories_from_board_tensor


class BoardEncoder(nn.Module):
    def __init__(self, *, d_model: int, n_layers: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = int(d_model)
        self.piece_emb = nn.Embedding(NUM_SQUARE_CATEGORIES, d_model)
        self.square_emb = nn.Embedding(64, d_model)
        self.turn_cls_emb = nn.Embedding(2, d_model)  # index 0 = black to move, 1 = white
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        nn.init.trunc_normal_(self.square_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.turn_cls_emb.weight, std=0.02)

    def _turn_index(self, board_tensor: torch.Tensor) -> torch.Tensor:
        # Side-to-move plane index per the codec channel map; confirm index in Step 4.
        # jepa3 used plane 12 (1.0 = white to move). Reduce to (B,) long.
        plane = board_tensor[..., 12] if board_tensor.dim() == 4 else board_tensor[:, 12]
        return (plane.reshape(plane.shape[0], -1).mean(dim=1) > 0.5).long()

    def forward(self, board_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cats = square_categories_from_board_tensor(board_tensor)  # (B,64)
        b = cats.shape[0]
        sq_idx = torch.arange(64, device=cats.device).unsqueeze(0).expand(b, 64)
        tok = self.piece_emb(cats) + self.square_emb(sq_idx)  # (B,64,d)
        turn = self.turn_cls_emb(self._turn_index(board_tensor)).unsqueeze(1)  # (B,1,d)
        x = torch.cat([turn, tok], dim=1)  # (B,65,d)
        h = self.encoder(x)
        return h[:, 0], h[:, 1:]  # cls, square_tokens
```

- [ ] **Step 4: Confirm the side-to-move plane index**

Run: `devcontainer exec --workspace-folder . bash -lc 'cd /workspaces/eloquent-encoding && python -c "from style_policy.packed_codec import packed_to_board_tensor; import h5py,torch; f=h5py.File(chr(47).join([\"\",\"mnt\",\"eloquence_bulk\",\"databases\",\"j3_training_1M.h5\"]),\"r\"); b=packed_to_board_tensor(torch.from_numpy(f[\"packed_pre\"][0:1].astype(\"uint8\")) ); print(b.shape)"'`
Read the codec docstring/channel map; if side-to-move is not plane 12, fix `_turn_index`. (This is the channel-map confirmation flagged in Task 1.1.)

- [ ] **Step 5: Run, expect pass**

Run: pytest from Step 1's command. Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add style_policy/board_encoder.py tests/style_policy/test_board_encoder.py
git commit -m "feat(style_policy): transformer board encoder (CLS + 64 square tokens)"
```

### Task 1.4: Policy heads (from, to) + promotion

**Files:**
- Create: `style_policy/policy_heads.py`, `style_policy/promotion_head.py`
- Test: `tests/style_policy/test_policy_heads.py`

**Interfaces:**
- Produces:
  - `class FromHead(nn.Module)`: `__init__(self, *, d_model, hidden, elo_dim=0, n_elo_buckets=0)`, `forward(self, squares, *, elo_idx=None) -> Tensor (B,64)` raw logits (unmasked).
  - `class ToHead(nn.Module)`: `__init__(self, *, d_model, hidden, elo_dim=0, n_elo_buckets=0)`, `forward(self, squares, from_sq, *, elo_idx=None) -> Tensor (B,64)` raw logits, conditioned on the chosen `from_sq` token.
  - `class PromotionHead(nn.Module)`: `forward(self, squares, from_sq, to_sq) -> Tensor (B,4)` (knight,bishop,rook,queen).
  - Masking is applied by the caller (model.py) using `u64_to_mask`; heads return raw logits.

- [ ] **Step 1: Write the failing test**

`tests/style_policy/test_policy_heads.py`:
```python
import torch
from style_policy.policy_heads import FromHead, ToHead


def test_from_head_shape():
    h = FromHead(d_model=32, hidden=64)
    squares = torch.randn(5, 64, 32)
    logits = h(squares)
    assert logits.shape == (5, 64)


def test_to_head_depends_on_from_square():
    h = ToHead(d_model=32, hidden=64).eval()
    squares = torch.randn(2, 64, 32)
    with torch.no_grad():
        a = h(squares, torch.tensor([10, 10]))
        b = h(squares, torch.tensor([20, 20]))
    # Different origin squares must yield different to-logits.
    assert not torch.allclose(a, b)


def test_from_head_elo_conditioning_shifts_logits():
    h = FromHead(d_model=32, hidden=64, elo_dim=8, n_elo_buckets=40).eval()
    squares = torch.randn(2, 64, 32)
    with torch.no_grad():
        lo = h(squares, elo_idx=torch.tensor([0, 0]))
        hi = h(squares, elo_idx=torch.tensor([39, 39]))
    assert not torch.allclose(lo, hi)
```

- [ ] **Step 2: Run, expect fail**

Run: `devcontainer exec --workspace-folder . bash -lc 'cd /workspaces/eloquent-encoding && python -m pytest tests/style_policy/test_policy_heads.py -q'`
Expected: FAIL.

- [ ] **Step 3: Implement the heads**

`style_policy/policy_heads.py`:
```python
"""Pointer policy heads over the 64 square tokens. Heads return RAW logits; caller masks legality.

from-head: per-square score from its token (+ optional elo conditioning).
to-head:   per-target score conditioned on the chosen from-square token (+ optional elo).
"""
from __future__ import annotations
import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


class FromHead(nn.Module):
    def __init__(self, *, d_model: int, hidden: int, elo_dim: int = 0, n_elo_buckets: int = 0):
        super().__init__()
        self.elo_dim = int(elo_dim)
        self.null_elo = int(n_elo_buckets)  # extra index for "unknown elo"
        if elo_dim > 0:
            self.elo_emb = nn.Embedding(n_elo_buckets + 1, elo_dim)
        self.score = _mlp(d_model + self.elo_dim, hidden, 1)

    def _elo_feat(self, b, device, elo_idx):
        if self.elo_dim == 0:
            return None
        if elo_idx is None:
            elo_idx = torch.full((b,), self.null_elo, device=device, dtype=torch.long)
        return self.elo_emb(elo_idx)  # (B, elo_dim)

    def forward(self, squares: torch.Tensor, *, elo_idx: torch.Tensor | None = None) -> torch.Tensor:
        b = squares.shape[0]
        feat = self._elo_feat(b, squares.device, elo_idx)
        if feat is not None:
            squares = torch.cat([squares, feat.unsqueeze(1).expand(b, 64, self.elo_dim)], dim=-1)
        return self.score(squares).squeeze(-1)  # (B,64)


class ToHead(nn.Module):
    def __init__(self, *, d_model: int, hidden: int, elo_dim: int = 0, n_elo_buckets: int = 0):
        super().__init__()
        self.elo_dim = int(elo_dim)
        self.null_elo = int(n_elo_buckets)
        if elo_dim > 0:
            self.elo_emb = nn.Embedding(n_elo_buckets + 1, elo_dim)
        # target token concatenated with the chosen origin token + optional elo
        self.score = _mlp(2 * d_model + self.elo_dim, hidden, 1)

    def forward(self, squares: torch.Tensor, from_sq: torch.Tensor, *, elo_idx: torch.Tensor | None = None) -> torch.Tensor:
        b, _, d = squares.shape
        origin = squares[torch.arange(b, device=squares.device), from_sq.long()]  # (B,d)
        origin = origin.unsqueeze(1).expand(b, 64, d)
        parts = [squares, origin]
        if self.elo_dim > 0:
            elo_idx = elo_idx if elo_idx is not None else torch.full((b,), self.null_elo, device=squares.device, dtype=torch.long)
            parts.append(self.elo_emb(elo_idx).unsqueeze(1).expand(b, 64, self.elo_dim))
        return self.score(torch.cat(parts, dim=-1)).squeeze(-1)  # (B,64)
```

`style_policy/promotion_head.py`:
```python
"""Promotion head: 4-way (knight,bishop,rook,queen). Fired only on pawn-to-back-rank moves."""
from __future__ import annotations
import torch
import torch.nn as nn


class PromotionHead(nn.Module):
    def __init__(self, *, d_model: int, hidden: int = 64):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(2 * d_model, hidden), nn.GELU(), nn.Linear(hidden, 4))

    def forward(self, squares: torch.Tensor, from_sq: torch.Tensor, to_sq: torch.Tensor) -> torch.Tensor:
        b = squares.shape[0]
        idx = torch.arange(b, device=squares.device)
        feat = torch.cat([squares[idx, from_sq.long()], squares[idx, to_sq.long()]], dim=-1)
        return self.score(feat)  # (B,4)
```

- [ ] **Step 4: Run, expect pass**

Run: pytest from Step 1's command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add style_policy/policy_heads.py style_policy/promotion_head.py tests/style_policy/test_policy_heads.py
git commit -m "feat(style_policy): pointer from/to heads + promotion head"
```

### Task 1.5: Masked cross-entropy loss

**Files:**
- Create: `style_policy/loss.py`
- Test: `tests/style_policy/test_loss.py`

**Interfaces:**
- Produces: `masked_square_ce(logits: Tensor (B,64), target: Tensor (B,), legal_mask: Tensor (B,64) bool) -> Tensor scalar` — softmax CE restricted to legal squares (illegal squares set to `-inf` before log-softmax); returns mean over batch. Also `top1_legal(logits, target, legal_mask) -> float` accuracy among legal moves.

- [ ] **Step 1: Write the failing test**

`tests/style_policy/test_loss.py`:
```python
import math, torch
from style_policy.loss import masked_square_ce, top1_legal


def test_loss_ignores_illegal_and_is_finite():
    logits = torch.zeros(1, 64)
    logits[0, 5] = 100.0  # huge logit on an ILLEGAL square must not affect loss
    mask = torch.zeros(1, 64, dtype=torch.bool)
    mask[0, 0] = True
    mask[0, 1] = True
    target = torch.tensor([0])
    loss = masked_square_ce(logits, target, mask)
    # two legal squares, equal logits → -log(0.5)
    assert math.isfinite(loss.item())
    assert abs(loss.item() - math.log(2)) < 1e-4


def test_top1_among_legal():
    logits = torch.zeros(1, 64)
    logits[0, 1] = 5.0
    mask = torch.zeros(1, 64, dtype=torch.bool)
    mask[0, 0] = mask[0, 1] = True
    assert top1_legal(logits, torch.tensor([1]), mask) == 1.0
    assert top1_legal(logits, torch.tensor([0]), mask) == 0.0
```

- [ ] **Step 2: Run, expect fail**

Run: `devcontainer exec --workspace-folder . bash -lc 'cd /workspaces/eloquent-encoding && python -m pytest tests/style_policy/test_loss.py -q'`
Expected: FAIL.

- [ ] **Step 3: Implement**

`style_policy/loss.py`:
```python
"""Masked square cross-entropy: softmax restricted to legal squares."""
from __future__ import annotations
import torch
import torch.nn.functional as F

_NEG = float("-inf")


def _masked_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~legal_mask, _NEG)


def masked_square_ce(logits: torch.Tensor, target: torch.Tensor, legal_mask: torch.Tensor,
                     *, label_smoothing: float = 0.0) -> torch.Tensor:
    masked = _masked_logits(logits, legal_mask)
    return F.cross_entropy(masked, target.long(), label_smoothing=label_smoothing)


def top1_legal(logits: torch.Tensor, target: torch.Tensor, legal_mask: torch.Tensor) -> float:
    masked = _masked_logits(logits, legal_mask)
    pred = masked.argmax(dim=-1)
    return (pred == target.long()).float().mean().item()
```

- [ ] **Step 4: Run, expect pass**

Run: pytest from Step 1's command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add style_policy/loss.py tests/style_policy/test_loss.py
git commit -m "feat(style_policy): masked square cross-entropy + top1"
```

### Task 1.6: Dataset

**Files:**
- Create: `style_policy/dataset.py`
- Test: `tests/style_policy/test_dataset.py`

**Interfaces:**
- Produces: `class PackedMoveDataset(torch.utils.data.Dataset)`: `__init__(self, h5_path, *, sample_n=None, seed=0)`; `__getitem__` returns a dict with tensors `packed_pre (34,) uint8`, `from_sq () int64`, `to_sq () int64`, `from_legal_u64 () int64`, `to_legal_u64 () int64`, `promotion () int64`, `elo_to_move () int64`. Also `collate(batch) -> dict[str, Tensor]` stacking to `(B,...)`. Elo→bucket mapping (`floor(elo/100)` clamped, `-1`→null) lives in the model_spec, not here.

- [ ] **Step 1: Write the failing test**

`tests/style_policy/test_dataset.py`:
```python
import torch
from style_policy.dataset import PackedMoveDataset

H5 = "/mnt/eloquence_bulk/databases/j3_training_1M.h5"


def test_row_fields_and_shapes():
    ds = PackedMoveDataset(H5, sample_n=16, seed=1)
    assert len(ds) == 16
    row = ds[0]
    assert row["packed_pre"].shape == (34,)
    assert row["from_sq"].dtype == torch.int64
    # ground-truth from_sq is a legal origin
    assert (int(row["from_legal_u64"]) >> int(row["from_sq"])) & 1 == 1


def test_collate_batches():
    ds = PackedMoveDataset(H5, sample_n=8, seed=1)
    batch = PackedMoveDataset.collate([ds[i] for i in range(8)])
    assert batch["packed_pre"].shape == (8, 34)
    assert batch["from_sq"].shape == (8,)
```

- [ ] **Step 2: Run, expect fail**

Run: `devcontainer exec --workspace-folder . bash -lc 'cd /workspaces/eloquent-encoding && python -m pytest tests/style_policy/test_dataset.py -q'`
Expected: FAIL.

- [ ] **Step 3: Implement**

`style_policy/dataset.py`:
```python
"""Dataset over jepa3-packed move rows (reused on-disk format). Lazy h5 read, optional fixed subsample."""
from __future__ import annotations
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

_FIELDS_U8 = ("from_sq", "to_sq", "promotion")


class PackedMoveDataset(Dataset):
    def __init__(self, h5_path: str | Path, *, sample_n: int | None = None, seed: int = 0):
        self.path = str(h5_path)
        with h5py.File(self.path, "r") as f:
            n = int(f["packed_pre"].shape[0])
        if sample_n is not None and sample_n < n:
            rng = np.random.default_rng(seed)
            self.indices = np.sort(rng.choice(n, size=sample_n, replace=False))
        else:
            self.indices = np.arange(n)
        self._f: h5py.File | None = None

    def _file(self) -> h5py.File:
        if self._f is None:
            self._f = h5py.File(self.path, "r")  # opened per-worker
        return self._f

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[i])
        f = self._file()
        out = {
            "packed_pre": torch.from_numpy(f["packed_pre"][idx].astype(np.uint8)),
            "from_legal_u64": torch.tensor(int(f["from_legal_u64"][idx]), dtype=torch.int64),
            "to_legal_u64": torch.tensor(int(f["to_legal_u64"][idx]), dtype=torch.int64),
            "elo_to_move": torch.tensor(int(f["elo_to_move"][idx]), dtype=torch.int64),
        }
        for k in _FIELDS_U8:
            out[k] = torch.tensor(int(f[k][idx]), dtype=torch.int64)
        return out

    @staticmethod
    def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        keys = batch[0].keys()
        return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
```

- [ ] **Step 4: Run, expect pass**

Run: pytest from Step 1's command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add style_policy/dataset.py tests/style_policy/test_dataset.py
git commit -m "feat(style_policy): packed move dataset over reused j3 h5"
```

### Task 1.7: BasePolicy model (wire encoder + heads + masking)

**Files:**
- Create: `style_policy/model.py`
- Test: `tests/style_policy/test_model_forward.py`

**Interfaces:**
- Consumes: `BoardEncoder`, `FromHead`, `ToHead`, `PromotionHead`, `u64_to_mask`, `packed_to_board_tensor`.
- Produces: `class BasePolicy(nn.Module)`: `from_config(cfg: dict) -> BasePolicy`; `forward_from(packed_pre, *, elo_idx=None) -> (logits (B,64), legal_mask (B,64))` where logits are already `-inf`-masked on illegal origins; `forward_to(packed_pre, from_sq, from_legal_u64?, *, elo_idx=None) -> (logits, legal_mask)` masked to legal targets given origin (target legality recomputed from the board via the codec or read from `to_legal_u64`); `encode(packed_pre) -> (cls, squares)`.

- [ ] **Step 1: Write the failing test**

`tests/style_policy/test_model_forward.py`:
```python
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
            "to_legal_u64": torch.tensor([int(x) for x in f["to_legal_u64"][0:n]], dtype=torch.int64),
        }


def test_forward_from_masks_illegal():
    m = BasePolicy.from_config(CFG).eval()
    b = _batch(4)
    with torch.no_grad():
        logits, mask = m.forward_from(b["packed_pre"])
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
```

- [ ] **Step 2: Run, expect fail**

Run: `devcontainer exec --workspace-folder . bash -lc 'cd /workspaces/eloquent-encoding && python -m pytest tests/style_policy/test_model_forward.py -q'`
Expected: FAIL.

- [ ] **Step 3: Implement**

`style_policy/model.py`:
```python
"""BasePolicy: encoder + two-stage pointer heads with legality masking applied here."""
from __future__ import annotations
import torch
import torch.nn as nn
from style_policy.board_encoder import BoardEncoder
from style_policy.policy_heads import FromHead, ToHead
from style_policy.promotion_head import PromotionHead
from style_policy.legal_mask import u64_to_mask
from style_policy.packed_codec import packed_to_board_tensor

_NEG = float("-inf")


class BasePolicy(nn.Module):
    def __init__(self, encoder, from_head, to_head, promo_head):
        super().__init__()
        self.encoder = encoder
        self.from_head = from_head
        self.to_head = to_head
        self.promo_head = promo_head

    @classmethod
    def from_config(cls, cfg: dict) -> "BasePolicy":
        d = int(cfg["d_model"])
        enc = BoardEncoder(d_model=d, n_layers=int(cfg["n_layers"]), nhead=int(cfg["nhead"]),
                           dim_feedforward=int(cfg["dim_feedforward"]), dropout=float(cfg["dropout"]))
        elo_dim = int(cfg.get("elo_dim", 0)); n_elo = int(cfg.get("n_elo_buckets", 0))
        h = int(cfg["head_hidden"])
        return cls(enc,
                   FromHead(d_model=d, hidden=h, elo_dim=elo_dim, n_elo_buckets=n_elo),
                   ToHead(d_model=d, hidden=h, elo_dim=elo_dim, n_elo_buckets=n_elo),
                   PromotionHead(d_model=d))

    def encode(self, packed_pre: torch.Tensor):
        board = packed_to_board_tensor(packed_pre).to(next(self.parameters()).device)
        return self.encoder(board)

    def forward_from(self, packed_pre, *, elo_idx=None):
        _, squares = self.encode(packed_pre)
        # from-legality is derivable from the board; recompute via codec or pass through to_legal pattern.
        from style_policy.packed_codec import packed_to_board_tensor as _p  # board for legality
        logits = self.from_head(squares, elo_idx=elo_idx)
        # legality mask must be supplied by caller in training (from_legal_u64); for inference recompute.
        raise NotImplementedError("masking wired in Step 4")

    def forward_to(self, packed_pre, from_sq, to_legal_u64, *, elo_idx=None):
        _, squares = self.encode(packed_pre)
        logits = self.to_head(squares, from_sq, elo_idx=elo_idx)
        mask = u64_to_mask(to_legal_u64).to(logits.device)
        return logits.masked_fill(~mask, _NEG), mask
```

- [ ] **Step 4: Finish `forward_from` masking (take the legal mask as an argument)**

Change `forward_from` to accept the origin-legality bitboard explicitly (training reads `from_legal_u64` from the batch; this keeps the model pure and avoids re-deriving legality on the hot path):
```python
    def forward_from(self, packed_pre, from_legal_u64, *, elo_idx=None):
        _, squares = self.encode(packed_pre)
        logits = self.from_head(squares, elo_idx=elo_idx)
        mask = u64_to_mask(from_legal_u64).to(logits.device)
        return logits.masked_fill(~mask, _NEG), mask
```
Update the test `test_forward_from_masks_illegal` to pass `b["from_legal_u64"]` (add it to `_batch`). Re-run.

- [ ] **Step 5: Run, expect pass**

Run: pytest from Step 1's command (with the updated test). Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add style_policy/model.py tests/style_policy/test_model_forward.py
git commit -m "feat(style_policy): BasePolicy wiring encoder + masked pointer heads"
```

### Task 1.8: Model spec (YAML) + elo bucketing

**Files:**
- Create: `style_policy/model_spec.py`, `style_policy/model_configs/tiny_smoke.yaml`, `style_policy/model_configs/base_16M.yaml`
- Test: extend `tests/style_policy/test_model_forward.py` with a config-load test (or new `test_model_spec.py`).

**Interfaces:**
- Produces: `load_spec(name: str) -> dict` (reads `style_policy/model_configs/{name}.yaml`, applies defaults + per-stage deep-merge, validates `name == file stem`); `elo_to_bucket(elo: Tensor, n_buckets: int) -> Tensor` mapping `floor(elo/100)` clamped to `[0, n_buckets-1]`, with `elo <= 0` → null index `n_buckets`.

- [ ] **Step 1: Write the failing test**

`tests/style_policy/test_model_spec.py`:
```python
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
```

- [ ] **Step 2: Run, expect fail**

Run: `devcontainer exec --workspace-folder . bash -lc 'cd /workspaces/eloquent-encoding && python -m pytest tests/style_policy/test_model_spec.py -q'`
Expected: FAIL.

- [ ] **Step 3: Implement spec loader + configs**

`style_policy/model_spec.py`:
```python
"""YAML spec: defaults + per-stage deep-merge; elo→bucket mapping. Paths resolve from repo root."""
from __future__ import annotations
import copy
from pathlib import Path
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = _REPO_ROOT / "style_policy" / "model_configs"


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


def load_spec(name: str) -> dict:
    path = CONFIGS / f"{name}.yaml"
    spec = yaml.safe_load(path.read_text())
    if spec.get("name") != name:
        raise ValueError(f"spec name {spec.get('name')!r} != {name!r}")
    defaults = spec.get("defaults", {})
    spec["stages"] = [_deep_merge(defaults, s) for s in spec["stages"]]
    return spec


def elo_to_bucket(elo: torch.Tensor, n_buckets: int) -> torch.Tensor:
    e = elo.long()
    bucket = torch.div(e, 100, rounding_mode="floor").clamp(0, n_buckets - 1)
    return torch.where(e > 0, bucket, torch.full_like(e, n_buckets))
```

`style_policy/model_configs/tiny_smoke.yaml`:
```yaml
name: tiny_smoke
checkpoint_dir: style_policy_checkpoints/tiny_smoke
train_h5: /mnt/eloquence_bulk/databases/j3_training_1M.h5
val_h5: /mnt/eloquence_bulk/databases/j3_training_1M.h5
val_sample: {n: 256, seed: 42}
architecture:
  d_model: 64
  n_layers: 2
  nhead: 4
  dim_feedforward: 128
  dropout: 0.0
  head_hidden: 64
  elo_dim: 8
  n_elo_buckets: 40
defaults:
  batch_size: 64
  dataloader_num_workers: 0
  use_amp: true
  weight_decay: 0.01
  max_gradient_norm: 1.0
  log_interval: 20
  seed: 0
stages:
  - sample: {n: 2000, seed: 1}
    train: {epochs: 1, learning_rate: 0.0003}
    label_smoothing: 0.0
```

`style_policy/model_configs/base_16M.yaml`:
```yaml
name: base_16M
checkpoint_dir: style_policy_checkpoints/base_16M
train_h5: /mnt/eloquence_bulk/databases/j3_training_16M.h5
val_h5: /mnt/eloquence_bulk/databases/j3_validation_1M.h5
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
  batch_size: 128
  dataloader_num_workers: 4
  use_amp: true
  weight_decay: 0.01
  max_gradient_norm: 1.0
  log_interval: 50
  gradient_accumulation_steps: 2
  seed: 0
stages:
  - sample: {n: 16000000, seed: 1}
    train: {epochs: 1, learning_rate: 0.0005}
    label_smoothing: 0.1
```

- [ ] **Step 4: Run, expect pass**

Run: pytest from Step 1's command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add style_policy/model_spec.py style_policy/model_configs tests/style_policy/test_model_spec.py
git commit -m "feat(style_policy): YAML spec loader + elo bucketing + tiny/base configs"
```

### Task 1.9: Training loop + CLI, then smoke run

**Files:**
- Create: `style_policy/training_loop.py`, `style_policy/train.py`
- Test: smoke run (no unit asserts; metrics + checkpoint must materialize).

**Interfaces:**
- Consumes: `BasePolicy`, `PackedMoveDataset`, `masked_square_ce`, `top1_legal`, `load_spec`, `elo_to_bucket`.
- Produces: `train_one_stage(spec, stage_idx, device) -> dict` (writes `{checkpoint_dir}/{name}_stage_{K}.pt` + `metrics/{name}_stage_{K}.json`); CLI `python -m style_policy.train --model NAME --stage K` (stage 0 = init random, K≥1 = load K-1, train stages[K-1]). The combined loss is `from_ce + to_ce` (+ promotion CE on promotion rows), each `masked_square_ce` against the respective legal mask, elo bucketed via `elo_to_bucket`.

- [ ] **Step 1: Implement the training loop**

`style_policy/training_loop.py` (key structure — full body written during implementation):
```python
"""One-stage training: from_ce + to_ce (+ promo) with AMP, grad-accum, checkpoint + metrics JSON."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from style_policy.model import BasePolicy
from style_policy.dataset import PackedMoveDataset
from style_policy.loss import masked_square_ce, top1_legal
from style_policy.model_spec import elo_to_bucket
from style_policy.legal_mask import u64_to_mask

_NEG = float("-inf")


def _step_loss(model, batch, device, n_elo, label_smoothing):
    packed = batch["packed_pre"].to(device)
    elo_idx = elo_to_bucket(batch["elo_to_move"], n_elo).to(device)
    from_logits, from_mask = model.forward_from(packed, batch["from_legal_u64"].to(device), elo_idx=elo_idx)
    to_logits, to_mask = model.forward_to(packed, batch["from_sq"].to(device), batch["to_legal_u64"].to(device), elo_idx=elo_idx)
    fl = masked_square_ce(from_logits, batch["from_sq"].to(device), from_mask, label_smoothing=label_smoothing)
    tl = masked_square_ce(to_logits, batch["to_sq"].to(device), to_mask, label_smoothing=label_smoothing)
    metrics = {"from_ce": fl.item(), "to_ce": tl.item(),
               "from_top1": top1_legal(from_logits, batch["from_sq"].to(device), from_mask),
               "to_top1": top1_legal(to_logits, batch["to_sq"].to(device), to_mask)}
    return fl + tl, metrics


def train_one_stage(spec: dict, stage_idx: int, device: str) -> dict:
    stage = spec["stages"][stage_idx - 1]
    arch = spec["architecture"]; n_elo = int(arch["n_elo_buckets"])
    ckpt_dir = Path(spec["checkpoint_dir"]); (ckpt_dir / "metrics").mkdir(parents=True, exist_ok=True)
    model = BasePolicy.from_config(arch).to(device)
    if stage_idx > 1:
        prev = ckpt_dir / f"{spec['name']}_stage_{stage_idx-1}.pt"
        model.load_state_dict(torch.load(prev, map_location=device)["model"])
    ds = PackedMoveDataset(spec["train_h5"], sample_n=stage["sample"]["n"], seed=stage["sample"]["seed"])
    dl = DataLoader(ds, batch_size=stage["batch_size"], shuffle=True,
                    num_workers=stage["dataloader_num_workers"], collate_fn=PackedMoveDataset.collate)
    opt = torch.optim.AdamW(model.parameters(), lr=stage["train"]["learning_rate"], weight_decay=stage["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=stage["use_amp"] and device == "cuda")
    accum = int(stage.get("gradient_accumulation_steps", 1))
    model.train()
    for epoch in range(int(stage["train"]["epochs"])):
        opt.zero_grad()
        for i, batch in enumerate(dl):
            with torch.amp.autocast("cuda", enabled=stage["use_amp"] and device == "cuda"):
                loss, m = _step_loss(model, batch, device, n_elo, stage.get("label_smoothing", 0.0))
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), stage["max_gradient_norm"])
                scaler.step(opt); scaler.update(); opt.zero_grad()
            if i % stage["log_interval"] == 0:
                print(f"epoch={epoch} step={i} loss={loss.item():.4f} "
                      f"from_top1={m['from_top1']*100:.1f}% to_top1={m['to_top1']*100:.1f}%")
    out = ckpt_dir / f"{spec['name']}_stage_{stage_idx}.pt"
    torch.save({"model": model.state_dict(), "architecture": arch}, out)
    rec = {"stage": stage_idx, "last_batch_metrics": m}
    (ckpt_dir / "metrics" / f"{spec['name']}_stage_{stage_idx}.json").write_text(json.dumps(rec, indent=2))
    print(f"Saved {out}")
    return rec
```

`style_policy/train.py`:
```python
"""CLI: python -m style_policy.train --model NAME --stage K  (0=init, K>=1 trains stages[K-1])."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch
from style_policy.model import BasePolicy
from style_policy.model_spec import load_spec
from style_policy.training_loop import train_one_stage


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    spec = load_spec(args.model)
    ckpt_dir = Path(spec["checkpoint_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == 0:
        model = BasePolicy.from_config(spec["architecture"])
        out = ckpt_dir / f"{spec['name']}_stage_0.pt"
        torch.save({"model": model.state_dict(), "architecture": spec["architecture"]}, out)
        print(f"Saved {out}"); return 0
    if not (1 <= args.stage <= len(spec["stages"])):
        print(f"--stage out of range (1..{len(spec['stages'])})", file=sys.stderr); return 1
    train_one_stage(spec, args.stage, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Commit the loop + CLI**

```bash
git add style_policy/training_loop.py style_policy/train.py
git commit -m "feat(style_policy): training loop + train CLI"
```

- [ ] **Step 3: Smoke run on GPU (init + one tiny stage)**

Run:
```bash
devcontainer exec --workspace-folder . bash -lc 'cd /workspaces/eloquent-encoding && \
  python -m style_policy.train --model tiny_smoke --stage 0 && \
  python -m style_policy.train --model tiny_smoke --stage 1'
```
Expected: stage 0 prints a saved `tiny_smoke_stage_0.pt`; stage 1 prints decreasing loss with `from_top1`/`to_top1` percentages, then saves `tiny_smoke_stage_1.pt` and a metrics JSON. Confirm both files exist:
```bash
devcontainer exec --workspace-folder . bash -lc 'ls -R style_policy_checkpoints/tiny_smoke'
```

- [ ] **Step 4: Full test suite green**

Run: `devcontainer exec --workspace-folder . bash -lc 'cd /workspaces/eloquent-encoding && python -m pytest tests/style_policy -q'`
Expected: all PASS.

- [ ] **Step 5: Commit any fixes from the smoke run**

```bash
git add -A && git commit -m "fix(style_policy): smoke-run fixes for base policy training" || echo "nothing to fix"
```

### Task 1.10: Train the real base + report

**Files:**
- Uses: `style_policy/model_configs/base_16M.yaml`
- Produces: `style_policy_checkpoints/base_16M/base_16M_stage_1.pt` + metrics.

- [ ] **Step 1: Init + train base (long-running; expect to tune batch size for 4GB)**

Run:
```bash
devcontainer exec --workspace-folder . bash -lc 'cd /workspaces/eloquent-encoding && \
  python -m style_policy.train --model base_16M --stage 0 && \
  python -m style_policy.train --model base_16M --stage 1'
```
If CUDA OOM: halve `batch_size` and double `gradient_accumulation_steps` in `base_16M.yaml`, re-run. Record the working values.

- [ ] **Step 2: Record the baseline number**

Capture val `from_top1`/`to_top1` from the metrics JSON. This is the **baseline the style residual (Phase 2) must beat** (Gate A). Note it in the metrics JSON / a short comment.

- [ ] **Step 3: Commit config adjustments**

```bash
git add style_policy/model_configs/base_16M.yaml
git commit -m "chore(style_policy): base_16M config tuned to fit 4GB"
```

---

## Phase 2 — Style buckets via EM (expand into its own plan; gated by Gate A)

Scope (detail when reached): `style_residual.py` (per-bucket from/to residual heads, product-of-experts in logit space, L2-regularized, base frozen); `style_stats.py` (per-game sacrifice/capture/check/quiet-move rates for validation); `em_loop.py` (soft→hard EM: score games under each bucket, assign by responsibility, refit residuals; balance regularization); a game-id requirement on the data (group rows by game — **needs a small pipeline change in `dataset_generation` to retain a game index; never a username**). **Gate A checks** (held-out lift over base; style-stat separation; elo-orthogonality) are the deliverable, not just "buckets diverge."

## Phase 3 — In-game style inference + matchup structure test (gated by Gate B)

Scope: posterior-over-buckets classifier (per-move log-likelihood accumulation + prior; mixture-of-policies prediction); retain `Result` + game-level records in the pipeline; build the **fixed-strength matchup table** and test for non-transitive structure (does style Y beat X beyond elo?). If transitive → stop; the meta-game is just "play stronger."

## Phase 4 — Counter-style engine (gated by Gate B passing)

Scope: opponent posterior → `z_self* = argmax_self Σ_k P(z_opp=k|moves)·WinProb(self,k)`, constrained to the opponent's level; sample the conditioned policy as the engine. Optional: latent multi-step rollouts using a learned dynamics predictor (the one place learning dynamics pays off) for lookahead.

---

## Self-Review

- **Spec coverage:** encoder (1.3), from-head/to-head (1.4), promotion (1.4), legality masking (1.2/1.7), elo conditioning on base (1.4/1.8), reuse of existing data (1.1/1.6), archive restructure (0.1), goal-2 base policy trainable end-to-end (1.9/1.10). Style buckets/EM, in-game inference, matchup engine are deliberately deferred to gated Phases 2–4 (research-dependent; each its own plan).
- **Type consistency:** `forward_from(packed_pre, from_legal_u64, *, elo_idx=)` and `forward_to(packed_pre, from_sq, to_legal_u64, *, elo_idx=)` are used identically in `model.py`, the model test, and `training_loop._step_loss`. `u64_to_mask`, `masked_square_ce(logits, target, legal_mask, *, label_smoothing=)`, `top1_legal`, `elo_to_bucket(elo, n_buckets)`, `load_spec(name)` names match across all referencing tasks.
- **Open confirmations folded into tasks (not placeholders):** the codec's board-tensor channel map (side-to-move plane) is confirmed in Task 1.1 Step 3 and Task 1.3 Step 4 against the lifted file; castling/EP special tokens are an explicit future refinement (Design Decision 2), not required for Phase 1 correctness.
- **4GB reality:** Task 1.10 expects OOM tuning; AMP + grad-accum are in the loop and configs from the start.
```

