# Bot rating via Maia2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Estimate an approximate Elo for one bot config by playing color-balanced games vs Maia2 (rapid) at several levels and fitting an anchored rating.

**Architecture:** Pure rating math (`rating.py`) + a `Maia2Bot` adapter on the existing `Player` seam + a runner that reuses `play_match`, tallies the bot's W/D/L per Maia level, and fits a single-parameter MLE Elo against the fixed Maia anchors.

**Tech Stack:** Python, python-chess, maia2 0.9, torch, the existing `style_policy/play.py` harness, pytest.

## Global Constraints

- **Execution environment:** run Python in the container —
  `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && PYTHONPATH=. <cmd>'`. Inside the container use `pytest` (NOT `python -m pytest`); scripts run as `python -m scripts.rate_bot`. Run `git` on the HOST. GPU (cuda) is available; maia2 0.9 is installed with weights cached at `maia2_models/` (gitignored).
- **Maia2 must SAMPLE its move** from `move_probs` (not argmax) so it plays at its nominal rating.
- All ratings are on the Maia/lichess-rapid scale (approximate, not FIDE).
- maia2 API: `model.from_pretrained(type="rapid", device="gpu"|"cpu")`; `inference.prepare()`;
  `move_probs, win_prob = inference.inference_each(model, prep, fen, elo_self, elo_oppo)` (`move_probs` = `{uci: prob}` over legal moves).
- Reuse `style_policy/play.py`: `Player`, `PolicyBot(checkpoint, elo, *, device, temperature, seed, opening_book, book_threshold)`, `play_match(white, black, n, *, max_plies) -> {white_wins, black_wins, draws, n}`. `OpeningBook.for_elo(book_dir, elo) -> OpeningBook|None`.
- Branch: `bot-rating-maia2`.
- Git commit footer (verbatim, both lines):
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01VMxeVCfznS5H68W5SGyXFC`

---

## File Structure

- `style_policy/rating.py` (+ `tests/style_policy/test_rating.py`) — CREATE: pure rating math.
- `style_policy/maia2_bot.py` (+ `tests/style_policy/test_maia2_bot.py`) — CREATE: `load_maia2` + `Maia2Bot(Player)`.
- `scripts/rate_bot.py` — CREATE: the runner (build bot, loop Maia levels color-balanced, fit + print).

---

## Task 1: Rating math (`style_policy/rating.py`)

**Files:**
- Create: `style_policy/rating.py`
- Test: `tests/style_policy/test_rating.py`

**Interfaces:**
- Produces: `expected_score(anchor, rating)->float`; `implied_rating(score, anchor, eps=1e-3)->float`;
  `score_ci(wins, draws, losses, z=1.96)->(score, lo, hi)`;
  `mle_rating(rows, iters=100)->(rating, se)` where `rows=[(anchor, n, score), …]`.

- [ ] **Step 1: Write the failing test**

Create `tests/style_policy/test_rating.py`:

```python
from style_policy.rating import expected_score, implied_rating, score_ci, mle_rating

def test_expected_score():
    assert abs(expected_score(1500, 1500) - 0.5) < 1e-9
    assert expected_score(1500, 1700) > 0.5   # stronger than the anchor
    assert expected_score(1900, 1500) < 0.5

def test_implied_rating():
    assert abs(implied_rating(0.5, 1500) - 1500) < 1e-6
    assert implied_rating(0.75, 1500) > 1500
    assert implied_rating(0.25, 1500) < 1500

def test_mle_recovers_known_rating():
    true_b = 1600
    rows = [(a, 1000, expected_score(a, true_b)) for a in (1100, 1300, 1500, 1700, 1900)]
    r, se = mle_rating(rows)
    assert abs(r - true_b) < 5
    assert se < 30

def test_mle_se_shrinks_with_n():
    small = [(a, 50, expected_score(a, 1600)) for a in (1300, 1500, 1700)]
    big = [(a, 5000, expected_score(a, 1600)) for a in (1300, 1500, 1700)]
    assert mle_rating(big)[1] < mle_rating(small)[1]

def test_score_ci_bounds():
    s, lo, hi = score_ci(50, 0, 50)
    assert abs(s - 0.5) < 1e-9 and 0.0 <= lo <= s <= hi <= 1.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && PYTHONPATH=. pytest tests/style_policy/test_rating.py -x -q'`
Expected: FAIL — `style_policy.rating` does not exist.

- [ ] **Step 3: Create `style_policy/rating.py`**

```python
"""Anchored Elo from match results. All ratings on the anchors' scale (Maia/lichess-rapid)."""
from __future__ import annotations
import math

_LN10_400 = math.log(10) / 400.0


def expected_score(anchor: float, rating: float) -> float:
    return 1.0 / (1.0 + 10 ** ((anchor - rating) / 400.0))


def implied_rating(score: float, anchor: float, eps: float = 1e-3) -> float:
    s = min(1 - eps, max(eps, score))
    return anchor + 400.0 * math.log10(s / (1 - s))


def score_ci(wins: int, draws: int, losses: int, z: float = 1.96):
    """(score, lo, hi) where score = (wins + 0.5*draws)/n, normal-approx CI clamped to [0,1]."""
    n = wins + draws + losses
    if n == 0:
        return 0.0, 0.0, 1.0
    score = (wins + 0.5 * draws) / n
    var = (wins * (1 - score) ** 2 + draws * (0.5 - score) ** 2 + losses * score ** 2) / (n * n)
    se = math.sqrt(var)
    return score, max(0.0, score - z * se), min(1.0, score + z * se)


def mle_rating(rows, iters: int = 100):
    """rows = [(anchor, n, score), …] -> (rating, se). Newton on the logistic log-likelihood."""
    rows = [(float(a), int(n), float(s)) for a, n, s in rows if n > 0]
    if not rows:
        raise ValueError("no games")
    anchors = [a for a, _, _ in rows]
    tot_n = sum(n for _, n, _ in rows)
    rating = sum(n * implied_rating(s, a) for a, n, s in rows) / tot_n  # init
    lo, hi = min(anchors) - 1200, max(anchors) + 1200
    for _ in range(iters):
        g = h = 0.0
        for a, n, s in rows:
            e = expected_score(a, rating)
            g += n * _LN10_400 * (s - e)
            h += -n * _LN10_400 ** 2 * e * (1 - e)
        if h == 0:
            break
        rating = min(hi, max(lo, rating - g / h))
        if abs(g / h) < 1e-6:
            break
    info = sum(n * _LN10_400 ** 2 * (lambda e: e * (1 - e))(expected_score(a, rating))
               for a, n, _ in rows)
    se = math.sqrt(1.0 / info) if info > 0 else float("inf")
    return rating, se
```

- [ ] **Step 4: Run the test**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && PYTHONPATH=. pytest tests/style_policy/test_rating.py -x -q'`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add style_policy/rating.py tests/style_policy/test_rating.py
git commit -m "feat(rating): anchored-Elo math (expected_score, implied_rating, score_ci, mle_rating)"
```

---

## Task 2: Maia2 adapter (`style_policy/maia2_bot.py`)

**Files:**
- Create: `style_policy/maia2_bot.py`
- Test: `tests/style_policy/test_maia2_bot.py`

**Interfaces:**
- Consumes: `style_policy.play.Player`; the maia2 API.
- Produces: `load_maia2(type="rapid", device="gpu") -> (model, prep)`;
  `Maia2Bot(Player)` with `__init__(self, model, prep, self_elo, opp_elo=None, seed=0)` and
  `choose_move(board) -> chess.Move` (samples from Maia2's legal move distribution).

- [ ] **Step 1: Write the failing (maia2-gated) test**

Create `tests/style_policy/test_maia2_bot.py`:

```python
import pytest, chess

pytest.importorskip("maia2")

@pytest.fixture(scope="module")
def maia():
    from style_policy.maia2_bot import load_maia2
    try:
        return load_maia2(device="cpu")   # weights cached at maia2_models/
    except Exception as e:
        pytest.skip(f"maia2 unavailable: {e}")

def test_maia2bot_returns_legal_move(maia):
    from style_policy.maia2_bot import Maia2Bot
    model, prep = maia
    bot = Maia2Bot(model, prep, self_elo=1500, seed=0)
    for fen in [chess.STARTING_FEN,
                "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"]:
        b = chess.Board(fen)
        assert bot.choose_move(b) in b.legal_moves

def test_maia2bot_seed_deterministic(maia):
    from style_policy.maia2_bot import Maia2Bot
    model, prep = maia
    a = Maia2Bot(model, prep, 1500, seed=7).choose_move(chess.Board())
    b = Maia2Bot(model, prep, 1500, seed=7).choose_move(chess.Board())
    assert a == b
```

- [ ] **Step 2: Run it to verify it fails**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && PYTHONPATH=. pytest tests/style_policy/test_maia2_bot.py -x -q'`
Expected: FAIL — `style_policy.maia2_bot` does not exist.

- [ ] **Step 3: Create `style_policy/maia2_bot.py`**

```python
"""Maia2 as an anchor opponent on the Player seam. Samples from Maia2's move distribution at a given
rating (sampling, not argmax, so it plays at its nominal level)."""
from __future__ import annotations
import random
import chess
from style_policy.play import Player


def load_maia2(type: str = "rapid", device: str = "gpu"):
    """Load a pretrained Maia2 model + the inference prep bundle. Falls back to CPU if GPU load fails."""
    from maia2 import model, inference
    try:
        m = model.from_pretrained(type=type, device=device)
    except Exception:
        m = model.from_pretrained(type=type, device="cpu")
    return m, inference.prepare()


class Maia2Bot(Player):
    def __init__(self, model, prep, self_elo: int, opp_elo: int | None = None, seed: int = 0):
        from maia2 import inference
        self._inference = inference
        self.model = model
        self.prep = prep
        self.self_elo = int(self_elo)
        self.opp_elo = int(opp_elo if opp_elo is not None else self_elo)
        self.rng = random.Random(seed)

    def choose_move(self, board: chess.Board) -> chess.Move:
        move_probs, _ = self._inference.inference_each(
            self.model, self.prep, board.fen(), self.self_elo, self.opp_elo)
        legal = {m.uci() for m in board.legal_moves}
        items = [(uci, p) for uci, p in move_probs.items() if uci in legal and p > 0]
        if not items:
            return self.rng.choice(list(board.legal_moves))
        ucis, weights = zip(*items)
        return chess.Move.from_uci(self.rng.choices(ucis, weights=weights, k=1)[0])
```

- [ ] **Step 4: Run the test**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && PYTHONPATH=. pytest tests/style_policy/test_maia2_bot.py -x -q'`
Expected: PASS (2 tests; uses the cached weights on CPU).

- [ ] **Step 5: Commit**

```bash
git add style_policy/maia2_bot.py tests/style_policy/test_maia2_bot.py
git commit -m "feat(rating): Maia2Bot anchor opponent on the Player seam"
```

---

## Task 3: Runner `scripts/rate_bot.py` + the rating run

**Files:**
- Create: `scripts/rate_bot.py`

**Interfaces:**
- Consumes: `PolicyBot`, `play_match`, `OpeningBook` (play/opening_book); `load_maia2`/`Maia2Bot` (Task 2); `implied_rating`/`score_ci`/`mle_rating` (Task 1).
- Produces: `bot_record_vs(bot, maia, games, max_plies) -> (wins, draws, losses)` and a CLI that prints the rating.

- [ ] **Step 1: Create `scripts/rate_bot.py`**

```python
#!/usr/bin/env python3
"""Rate one bot config by playing color-balanced games vs Maia2 (rapid) at several levels.

Run: python -m scripts.rate_bot [--checkpoint ...] [--elo 1500] [--temperature 0.1]
                                [--levels 1100 1300 1500 1700 1900] [--games-per-level 100]
"""
from __future__ import annotations
import argparse, json, time
from style_policy.play import PolicyBot, play_match
from style_policy.opening_book import OpeningBook
from style_policy.maia2_bot import load_maia2, Maia2Bot
from style_policy.rating import implied_rating, score_ci, mle_rating


def bot_record_vs(bot, maia, games: int, max_plies: int):
    """Bot's (wins, draws, losses) over `games`, split half as White and half as Black."""
    half = games // 2
    a = play_match(bot, maia, half, max_plies=max_plies)            # bot = White
    b = play_match(maia, bot, games - half, max_plies=max_plies)    # bot = Black
    wins = a["white_wins"] + b["black_wins"]
    losses = a["black_wins"] + b["white_wins"]
    draws = a["draws"] + b["draws"]
    return wins, draws, losses


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="style_policy_checkpoints/base_64M/base_64M_stage_1.pt")
    ap.add_argument("--elo", type=int, default=1500)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--book", action="store_true", default=True)
    ap.add_argument("--no-book", dest="book", action="store_false")
    ap.add_argument("--book-dir", default="/mnt/eloquence_bulk/databases/opening_book")
    ap.add_argument("--levels", type=int, nargs="+", default=[1100, 1300, 1500, 1700, 1900])
    ap.add_argument("--games-per-level", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    book = OpeningBook.for_elo(a.book_dir, a.elo) if a.book else None
    if a.book and book is None:
        print(f"(no opening book for elo {a.elo} in {a.book_dir}; playing without book)")
    bot = PolicyBot(a.checkpoint, a.elo, device=a.device, temperature=a.temperature,
                    seed=a.seed, opening_book=book)
    maia_device = "gpu" if str(a.device).startswith("cuda") else "cpu"
    model, prep = load_maia2("rapid", device=maia_device)

    print(f"rating bot: {a.checkpoint} @ elo {a.elo}, T={a.temperature}, book={'on' if book else 'off'}")
    print(f"vs Maia2 rapid {a.levels}, {a.games_per_level} games/level (color-balanced)\n")
    print(f"{'maia':>5} {'W':>4} {'D':>4} {'L':>4} {'score':>6}  implied (95% CI)")
    rows, per_level, t0 = [], [], time.time()
    for R in a.levels:
        maia = Maia2Bot(model, prep, self_elo=R, seed=a.seed + R)
        w, d, l = bot_record_vs(bot, maia, a.games_per_level, a.max_plies)
        score, lo, hi = score_ci(w, d, l)
        imp = implied_rating(score, R)
        rows.append((R, w + d + l, score))
        per_level.append({"level": R, "w": w, "d": d, "l": l, "score": score,
                          "implied": imp, "implied_lo": implied_rating(lo, R),
                          "implied_hi": implied_rating(hi, R)})
        print(f"{R:>5} {w:>4} {d:>4} {l:>4} {score:>6.3f}  {imp:6.0f} "
              f"[{implied_rating(lo, R):.0f}, {implied_rating(hi, R):.0f}]", flush=True)

    rating, se = mle_rating(rows)
    scores = [s for _, _, s in rows]
    mono = all(scores[i] >= scores[i + 1] - 1e-9 for i in range(len(scores) - 1))
    print(f"\n==> bot rating ≈ {rating:.0f}  (95% CI ±{1.96 * se:.0f})   "
          f"[Maia/lichess-rapid scale]   monotonic={mono}")
    print(f"    {sum(n for _, n, _ in rows)} games in {time.time() - t0:.0f}s")
    if a.out:
        json.dump({"checkpoint": a.checkpoint, "elo": a.elo, "temperature": a.temperature,
                   "rating": rating, "se": se, "per_level": per_level}, open(a.out, "w"), indent=2)
        print(f"    wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke-run the runner (tiny, gated on Maia2)**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && PYTHONPATH=. python -m scripts.rate_bot --levels 1500 --games-per-level 2 --device cuda'`
Expected: completes; prints one table row and a finite `bot rating ≈ …`. (Confirms the bot + Maia2 + play loop + rating fit all wire together.)

- [ ] **Step 3: Commit the runner**

```bash
git add scripts/rate_bot.py
git commit -m "feat(rating): rate_bot runner (color-balanced games vs Maia2 -> anchored Elo)"
```

- [ ] **Step 4: Run the real rating (base_64M @ 1500) + report**

Run: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && PYTHONPATH=. python -m scripts.rate_bot --device cuda --out /workspaces/eloquent-encoding/.rate_base64M_1500.json 2>&1 | grep -vE "Warning|warn"'`
Expected: a per-level table (scores falling as the Maia level rises), a headline `bot rating ≈ …` with CI, `monotonic=True`, in a few minutes. Paste the table + rating into the report. The `.json` is a data artifact (gitignored — add `.rate_*.json` to `.gitignore`); do not commit it.

- [ ] **Step 5: Commit (gitignore the results artifact; no code change otherwise)**

```bash
grep -q '.rate_.*json' .gitignore || echo '.rate_*.json' >> .gitignore
git add .gitignore
git commit -m "chore: gitignore rate_bot result artifacts"
```

---

## Self-review notes

- **Spec coverage:** rating math incl. MLE + CI (T1); `Maia2Bot` sampling-not-argmax on the Player seam + `load_maia2` (T2); color-balanced runner reusing `play_match`, per-level implied + global MLE rating, monotonicity check, the real run (T3). Tests: pure rating (T1), maia2-gated adapter (T2), runner smoke + real run (T3). Existing `play.py` tests untouched.
- **Naming consistency:** `expected_score`/`implied_rating`/`score_ci`/`mle_rating`, `load_maia2`/`Maia2Bot(model, prep, self_elo, opp_elo, seed)`, `bot_record_vs`, rows `(anchor, n, score)` used identically across tasks.
- **Known follow-ups (out of scope):** the elo-sweep calibration curve (loop `--elo`); blitz Maia; multi-bot Ordo; lichess-bot.
```
