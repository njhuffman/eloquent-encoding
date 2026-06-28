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
