# Expectimax (WDL) search bot — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A depth-limited top-K expectimax bot that proposes/opponent-models with the policy and scores leaves with the WDL value head, rated at several depths vs Maia2.

**Architecture:** A pure `expectimax` over generic nodes + an `ExpectimaxBot(Player)` that wires chess (top-K policy children, bot-POV WDL leaf value) into it on `wdl_16M`, plus a runner that rates each depth vs Maia2 by reusing the existing rating tool.

**Tech Stack:** Python, python-chess, torch, the existing `style_policy/play.py` + Maia2 rating tooling, pytest.

## Global Constraints

- **Execution environment:** run Python in the container —
  `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && PYTHONPATH=. <cmd>'`. Inside the container use `pytest` (NOT `python -m pytest`); scripts run as `python -m scripts.rate_search`. Run `git` on the HOST. GPU (cuda) available.
- **Use `wdl_16M`** (`style_policy_checkpoints/wdl_16M/wdl_16M_stage_1.pt`) — it has the trained value head — for policy AND value.
- Leaf objective = bot-perspective **WDL escore** = `P(win)+0.5·P(draw)` (STM), flipped to bot POV; terminal positions scored exactly (1 win / 0.5 draw / 0 loss).
- Opponent chance-nodes use the **same model at the bot's elo** (single-elo); search is **deterministic** (top-K, no sampling).
- Legality masks: `u64_to_mask(torch.from_numpy(np.array([u64], dtype=np.uint64)).to(torch.int64))` (the uint64→int64 reinterpret avoids overflow when bit 63 is set).
- Reuse: `style_policy/play.py` (`Player`), `style_policy/board_encode.py` (`board_to_packed`, `legal_from_u64`, `legal_to_u64`), `style_policy/legal_mask.u64_to_mask`, `style_policy/model.py` (`encode`, `from_head`, `to_head`, `value_head`); Maia2 rating tool (`load_maia2`, `Maia2Bot`, `bot_record_vs`, `mle_rating`); `style_policy/opening_book.py` (`OpeningBook.for_elo`).
- Branch: `expectimax-bot`.
- Git commit footer (verbatim, both lines):
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01VMxeVCfznS5H68W5SGyXFC`

---

## File Structure

- `style_policy/search.py` (+ `tests/style_policy/test_search.py`) — CREATE: pure `expectimax`.
- `style_policy/search_bot.py` (+ `tests/style_policy/test_search_bot.py`) — CREATE: `policy_topk` + `ExpectimaxBot(Player)`.
- `scripts/rate_search.py` — CREATE: depth-sweep runner (rate each depth vs Maia2).

---

## Task 1: Pure expectimax (`style_policy/search.py`)

**Files:**
- Create: `style_policy/search.py`
- Test: `tests/style_policy/test_search.py`

**Interfaces:**
- Produces: `expectimax(node, depth, expand, leaf_value, is_max_node) -> (value: float, best_move)`.
  `expand(node) -> list[(move, child, prob)]`; `leaf_value(node) -> float`; `is_max_node(node) -> bool`.
  Max node returns the best child's value + its move; chance node returns the prob-weighted (renormalized)
  expectation + `None`; depth≤0 or empty `expand` returns `(leaf_value(node), None)`.

- [ ] **Step 1: Write the failing test**

Create `tests/style_policy/test_search.py`:

```python
from style_policy.search import expectimax

# toy tree: root (max) -> A (chance), B (chance); A -> Ax,Ay ; B -> Bz
_TREE = {"root": [("a", "A", 0.5), ("b", "B", 0.5)],
         "A": [("x", "Ax", 0.7), ("y", "Ay", 0.3)],
         "B": [("z", "Bz", 1.0)]}
_LEAF = {"Ax": 1.0, "Ay": 0.0, "Bz": 0.5}
_ISMAX = {"root": True, "A": False, "B": False}
_expand = lambda n: _TREE.get(n, [])
_leaf = lambda n: _LEAF.get(n, 0.0)
_ismax = lambda n: _ISMAX.get(n, True)

def test_max_then_chance():
    # depth 2: A=0.7*1+0.3*0=0.7 ; B=1.0*0.5=0.5 ; root max -> 0.7 via "a"
    v, m = expectimax("root", 2, _expand, _leaf, _ismax)
    assert m == "a" and abs(v - 0.7) < 1e-9

def test_depth_limit_stops_at_one():
    # depth 1: children A,B evaluated as leaves (not in _LEAF -> 0.0) -> v 0, move first
    v, m = expectimax("root", 1, _expand, _leaf, _ismax)
    assert v == 0.0 and m == "a"

def test_depth_zero_is_leaf():
    assert expectimax("Ax", 0, _expand, _leaf, _ismax) == (1.0, None)

def test_terminal_empty_expand_is_leaf():
    # Bz has no children -> leaf value even at depth>0
    assert expectimax("Bz", 5, _expand, _leaf, _ismax) == (0.5, None)

def test_chance_renormalizes():
    tree = {"r": [("a", "x", 1.0), ("b", "y", 3.0)]}   # unnormalized weights
    v, _ = expectimax("r", 1, lambda n: tree.get(n, []),
                      lambda n: {"x": 0.0, "y": 1.0}.get(n, 0.0), lambda n: False)
    assert abs(v - 0.75) < 1e-9   # 3/(1+3)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && PYTHONPATH=. pytest tests/style_policy/test_search.py -x -q'`
Expected: FAIL — `style_policy.search` does not exist.

- [ ] **Step 3: Create `style_policy/search.py`**

```python
"""Pure depth-limited expectimax over generic nodes (no chess/torch deps)."""
from __future__ import annotations


def expectimax(node, depth, expand, leaf_value, is_max_node):
    """Return (value, best_move). Max node -> best child's value + its move; chance node ->
    probability-weighted (renormalized) expectation + None; depth<=0 or no children -> leaf."""
    if depth <= 0:
        return leaf_value(node), None
    children = expand(node)  # [(move, child, prob), ...]
    if not children:
        return leaf_value(node), None
    if is_max_node(node):
        best_v, best_m = float("-inf"), None
        for mv, child, _ in children:
            v, _ = expectimax(child, depth - 1, expand, leaf_value, is_max_node)
            if v > best_v:
                best_v, best_m = v, mv
        return best_v, best_m
    total = sum(p for _, _, p in children) or 1.0
    v = 0.0
    for _, child, p in children:
        cv, _ = expectimax(child, depth - 1, expand, leaf_value, is_max_node)
        v += (p / total) * cv
    return v, None
```

- [ ] **Step 4: Run the test**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && PYTHONPATH=. pytest tests/style_policy/test_search.py -x -q'`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add style_policy/search.py tests/style_policy/test_search.py
git commit -m "feat(search): pure depth-limited expectimax"
```

---

## Task 2: `policy_topk` + `ExpectimaxBot` (`style_policy/search_bot.py`)

**Files:**
- Create: `style_policy/search_bot.py`
- Test: `tests/style_policy/test_search_bot.py`

**Interfaces:**
- Consumes: `expectimax` (Task 1); model heads; `board_to_packed`/`legal_*`/`u64_to_mask`; `Player`.
- Produces: `policy_topk(model, board, elo_idx, k, device) -> list[(chess.Move, float)]` (descending prob,
  one entry per distinct legal from→to, promotions represented by the queen move);
  `ExpectimaxBot(Player)` with `__init__(self, checkpoint, elo, depth, *, width=4, device="cpu", seed=0, opening_book=None, book_threshold=0.01)` and `choose_move(board)->chess.Move`.

- [ ] **Step 1: Write the failing (wdl_16M-gated) test**

Create `tests/style_policy/test_search_bot.py`:

```python
import os, pytest, chess

CKPT = "style_policy_checkpoints/wdl_16M/wdl_16M_stage_1.pt"
pytestmark = pytest.mark.skipif(not os.path.exists(CKPT), reason="wdl_16M checkpoint missing")

@pytest.fixture(scope="module")
def bot0():
    from style_policy.search_bot import ExpectimaxBot
    return ExpectimaxBot(CKPT, elo=1500, depth=0, width=4, device="cpu", seed=0)

def test_policy_topk_shape(bot0):
    from style_policy.search_bot import policy_topk
    b = chess.Board()
    out = policy_topk(bot0.model, b, bot0._elo_idx, 4, "cpu")
    assert 1 <= len(out) <= 4
    moves = [m for m, _ in out]
    assert all(m in b.legal_moves for m in moves)
    probs = [p for _, p in out]
    assert probs == sorted(probs, reverse=True) and all(0 < p <= 1 for p in probs)

def test_depth0_is_policy_argmax(bot0):
    from style_policy.search_bot import policy_topk
    b = chess.Board()
    assert bot0.choose_move(b) == policy_topk(bot0.model, b, bot0._elo_idx, 4, "cpu")[0][0]

def test_depths_return_legal_and_deterministic():
    from style_policy.search_bot import ExpectimaxBot
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    for d in (1, 2):
        m1 = ExpectimaxBot(CKPT, 1500, d, width=4, device="cpu").choose_move(b.copy())
        m2 = ExpectimaxBot(CKPT, 1500, d, width=4, device="cpu").choose_move(b.copy())
        assert m1 in b.legal_moves and m1 == m2  # legal + deterministic
```

- [ ] **Step 2: Run it to verify it fails**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && PYTHONPATH=. pytest tests/style_policy/test_search_bot.py -x -q'`
Expected: FAIL — `style_policy.search_bot` does not exist.

- [ ] **Step 3: Create `style_policy/search_bot.py`**

```python
"""Expectimax search bot: policy proposes/opponent-models, WDL value head scores leaves.
Built on wdl_16M (the checkpoint with a trained value head)."""
from __future__ import annotations
import random
import numpy as np
import torch
import chess
from style_policy.model import BasePolicy
from style_policy.model_spec import elo_to_bucket
from style_policy.board_encode import board_to_packed, legal_from_u64, legal_to_u64
from style_policy.legal_mask import u64_to_mask
from style_policy.play import Player
from style_policy.search import expectimax

_NEG = float("-inf")


def _mask(u64: int) -> torch.Tensor:
    return u64_to_mask(torch.from_numpy(np.array([u64], dtype=np.uint64)).to(torch.int64))


def _escore(wdl_logits: torch.Tensor) -> float:
    p = torch.softmax(wdl_logits, dim=-1)[0]   # [loss, draw, win]
    return float(p[2] + 0.5 * p[1])


@torch.no_grad()
def policy_topk(model, board: chess.Board, elo_idx, k: int, device: str):
    """Top-k legal moves by joint P(from)*P(to|from), descending. One entry per distinct (from,to);
    a promotion (from,to) is represented by its queen move (the policy is from/to only)."""
    pk = torch.from_numpy(board_to_packed(board)[None]).to(device)
    _, squares = model.encode(pk)
    fl = model.from_head(squares, elo_idx=elo_idx)
    fprob = torch.softmax(fl.masked_fill(~_mask(legal_from_u64(board)).to(fl.device), _NEG), dim=-1)[0]
    to_cache: dict[int, torch.Tensor] = {}
    best: dict[tuple, tuple] = {}
    for mv in board.legal_moves:
        key = (mv.from_square, mv.to_square)
        if key in best:
            continue
        f = mv.from_square
        if f not in to_cache:
            tl = model.to_head(squares, torch.tensor([f], device=device), elo_idx=elo_idx)
            to_cache[f] = torch.softmax(tl.masked_fill(~_mask(legal_to_u64(board, f)).to(tl.device), _NEG), dim=-1)[0]
        p = float(fprob[f] * to_cache[f][mv.to_square])
        cm = mv if mv.promotion in (None, chess.QUEEN) else chess.Move(f, mv.to_square, promotion=chess.QUEEN)
        best[key] = (cm, p)
    return sorted(best.values(), key=lambda x: -x[1])[:k]


class ExpectimaxBot(Player):
    def __init__(self, checkpoint, elo, depth, *, width=4, device="cpu", seed=0,
                 opening_book=None, book_threshold=0.01):
        ck = torch.load(checkpoint, map_location=device)
        self.model = BasePolicy.from_config(ck["architecture"]).to(device)
        _loaded = self.model.load_state_dict(ck["model"], strict=False)
        assert not _loaded.unexpected_keys and all(k.startswith("value_head") for k in _loaded.missing_keys), \
            f"checkpoint mismatch: unexpected={_loaded.unexpected_keys} missing={_loaded.missing_keys}"
        self.model.eval()
        self.device = device
        self.depth = int(depth)
        self.width = int(width)
        n_elo = int(ck["architecture"]["n_elo_buckets"])
        self._elo_idx = elo_to_bucket(torch.tensor([int(elo)]), n_elo).to(device)
        self.opening_book = opening_book
        self.book_threshold = float(book_threshold)
        self._book_rng = random.Random(seed)

    @torch.no_grad()
    def choose_move(self, board: chess.Board) -> chess.Move:
        if self.opening_book is not None:
            mv = self.opening_book.lookup(board, self.book_threshold, self._book_rng)
            if mv is not None:
                return mv
        if self.depth == 0:
            return policy_topk(self.model, board, self._elo_idx, self.width, self.device)[0][0]
        bot_color = board.turn

        def expand(b):
            if b.is_game_over():
                return []
            out = []
            for mv, p in policy_topk(self.model, b, self._elo_idx, self.width, self.device):
                c = b.copy(stack=False); c.push(mv)
                out.append((mv, c, p))
            return out

        def is_max(b):
            return b.turn == bot_color

        def leaf_value(b):
            if b.is_game_over():
                o = b.outcome()
                if o is None or o.winner is None:
                    return 0.5
                return 1.0 if o.winner == bot_color else 0.0
            pk = torch.from_numpy(board_to_packed(b)[None]).to(self.device)
            e = _escore(self.model.value_head(self.model.encode(pk)[0], elo_idx=self._elo_idx))
            return e if b.turn == bot_color else 1.0 - e

        _, move = expectimax(board, self.depth, expand, leaf_value, is_max)
        return move
```

- [ ] **Step 4: Run the test**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && PYTHONPATH=. pytest tests/style_policy/test_search_bot.py -x -q'`
Expected: PASS (3 tests, using the wdl_16M checkpoint on CPU).

- [ ] **Step 5: Commit**

```bash
git add style_policy/search_bot.py tests/style_policy/test_search_bot.py
git commit -m "feat(search): policy_topk + ExpectimaxBot (WDL leaf eval on wdl_16M)"
```

---

## Task 3: Depth-sweep runner `scripts/rate_search.py` + the run

**Files:**
- Create: `scripts/rate_search.py`

**Interfaces:**
- Consumes: `ExpectimaxBot` (Task 2); `OpeningBook`; `load_maia2`/`Maia2Bot` + `bot_record_vs` + `mle_rating`.

- [ ] **Step 1: Create `scripts/rate_search.py`**

```python
#!/usr/bin/env python3
"""Rate the expectimax bot at several search depths vs Maia2 (rapid)."""
from __future__ import annotations
import argparse, json, time
from style_policy.opening_book import OpeningBook
from style_policy.maia2_bot import load_maia2, Maia2Bot
from style_policy.rating import mle_rating
from style_policy.search_bot import ExpectimaxBot
from scripts.rate_bot import bot_record_vs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="style_policy_checkpoints/wdl_16M/wdl_16M_stage_1.pt")
    ap.add_argument("--elo", type=int, default=1500)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--depths", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--levels", type=int, nargs="+", default=[1100, 1300, 1500, 1700, 1900])
    ap.add_argument("--games-per-level", type=int, default=25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--book", action="store_true", default=True)
    ap.add_argument("--no-book", dest="book", action="store_false")
    ap.add_argument("--book-dir", default="/mnt/eloquence_bulk/databases/opening_book")
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    maia_dev = "gpu" if str(a.device).startswith("cuda") else "cpu"
    model, prep = load_maia2("rapid", device=maia_dev)
    book = OpeningBook.for_elo(a.book_dir, a.elo) if a.book else None
    print(f"expectimax {a.checkpoint} @ elo {a.elo}, width {a.width}, book={'on' if book else 'off'}")
    print(f"depths {a.depths} vs Maia {a.levels}, {a.games_per_level} games/level\n")
    print(f"{'depth':>5} {'rating':>7} {'±95%':>5}   per-Maia score [{' '.join(str(L) for L in a.levels)}]")
    results, t0 = [], time.time()
    for depth in a.depths:
        bot = ExpectimaxBot(a.checkpoint, a.elo, depth, width=a.width, device=a.device,
                            seed=a.seed, opening_book=book)
        rows, scores = [], []
        for R in a.levels:
            maia = Maia2Bot(model, prep, self_elo=R, seed=a.seed + R)
            w, d, l = bot_record_vs(bot, maia, a.games_per_level, a.max_plies)
            s = (w + 0.5 * d) / (w + d + l)
            rows.append((R, w + d + l, s)); scores.append(s)
        rating, se = mle_rating(rows)
        results.append({"depth": depth, "rating": rating, "se": se, "scores": scores})
        print(f"{depth:>5} {rating:>7.0f} {1.96*se:>5.0f}   " + "  ".join(f"{x:.2f}" for x in scores), flush=True)
    print(f"\n{time.time()-t0:.0f}s total")
    if a.out:
        json.dump({"checkpoint": a.checkpoint, "elo": a.elo, "width": a.width, "results": results},
                  open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke-run (tiny)**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && PYTHONPATH=. python -m scripts.rate_search --depths 0 1 --levels 1500 --games-per-level 2 --device cuda 2>&1 | grep -vE "Warning|warn"'`
Expected: completes; prints two rows (depth 0 and 1) with finite ratings. (Confirms ExpectimaxBot + Maia2 + rating wire together at depth 0 and a search depth.)

- [ ] **Step 3: Commit the runner**

```bash
git add scripts/rate_search.py
git commit -m "feat(search): rate_search depth-sweep runner (expectimax vs Maia2)"
```

- [ ] **Step 4: Run the depth sweep + report**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && PYTHONPATH=. python -m scripts.rate_search --device cuda --out /workspaces/eloquent-encoding/.rate_search_wdl16M.json 2>&1 | grep -vE "Warning|warn"'`
Expected: a `depth → rating ±CI` table (depths 0,1,2,3). depth 3 is slow (~tens of minutes) — let it finish; rows stream per depth. Paste the full table into your report. The `.json` is a gitignored artifact (`.rate_*.json` already ignored) — do not commit it. If depth 3 is impractically slow, report depths 0–2 and note depth 3 was cut.

- [ ] **Step 5: Done (no code change)**

No commit needed beyond Step 3; the produced JSON is a data artifact. If the run surfaced a bug, fix under TDD and amend Step 3.

---

## Self-review notes

- **Spec coverage:** pure expectimax with max/chance/leaf/depth (T1); `policy_topk` top-K + `ExpectimaxBot` (depth-0 argmax baseline, depth≥1 expectimax, bot-POV WDL leaf incl. terminal, opening-book, deterministic) on wdl_16M (T2); depth-sweep rating vs Maia2 reusing the tool + the real run (T3). Tests: pure search toy-tree, gated policy_topk/bot, runner smoke + real run.
- **Naming consistency:** `expectimax(node, depth, expand, leaf_value, is_max_node)`, `policy_topk(model, board, elo_idx, k, device)`, `ExpectimaxBot(checkpoint, elo, depth, *, width, device, seed, opening_book, book_threshold)`, `_escore`/`_mask`, reused `bot_record_vs`/`mle_rating`/`Maia2Bot`/`load_maia2`/`OpeningBook.for_elo` — consistent across tasks.
- **Known follow-ups (out of scope):** batched leaf eval; depth-varying width; minimax opponent; base_64M policy.
```
