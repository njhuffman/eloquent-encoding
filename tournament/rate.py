"""Joint rating + calibration. Reads all of results.jsonl, fits one Bradley-Terry/Elo MLE over the
whole field (gradient ascent of the logistic log-likelihood, draws=0.5), then affine-calibrates the
fitted Elo to the known lichess-rapid ratings of the anchor bots (ref_elo). Bootstrap CIs. Prints a
ranked table on the lichess-rapid scale."""
from __future__ import annotations
import argparse, json, random
import numpy as np
from tournament.bots import load_registry


def _fit(games, n, idx, iters=4000, lr=6.0):
    r = np.zeros(n)
    for _ in range(iters):
        grad = np.zeros(n)
        for i, j, s in games:
            ea = 1.0 / (1.0 + 10 ** ((r[j] - r[i]) / 400.0))
            g = s - ea
            grad[i] += g; grad[j] -= g
        r += lr * grad / max(1, len(games) / n)
        r -= r.mean()
    return r


def _calibrate(r, anchors):
    """anchors: list of (index, ref_elo). Least-squares affine fit of ref ~ slope*r + intercept."""
    if len(anchors) < 2:
        return r + 1500.0  # no calibration possible; arbitrary offset
    xs = np.array([r[i] for i, _ in anchors]); ys = np.array([e for _, e in anchors])
    slope, intercept = np.polyfit(xs, ys, 1)
    return slope * r + intercept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="tournament/bots.yaml")
    ap.add_argument("--results", default="tournament/results.jsonl")
    ap.add_argument("--bootstrap", type=int, default=60)
    a = ap.parse_args()

    reg = load_registry(a.registry)
    ids = [e["id"] for e in reg]; idx = {b: i for i, b in enumerate(ids)}
    ref = {e["id"]: e.get("ref_elo") for e in reg}
    rows = [json.loads(l) for l in open(a.results)]
    games, gcount = [], {b: 0 for b in ids}
    for g in rows:
        if g["a"] not in idx or g["b"] not in idx:
            continue
        i, j = idx[g["a"]], idx[g["b"]]
        s = 1.0 if g["result"] == "a" else (0.0 if g["result"] == "b" else 0.5)
        games.append((i, j, s)); gcount[g["a"]] += 1; gcount[g["b"]] += 1
    if not games:
        print("no games in results yet"); return 0
    anchors = [(idx[b], ref[b]) for b in ids if ref[b] is not None]

    r = _fit(games, len(ids), idx)
    cal = _calibrate(r, anchors)

    # bootstrap CIs over resampled games
    boot = np.zeros((a.bootstrap, len(ids)))
    rng = random.Random(0)
    for b in range(a.bootstrap):
        samp = [games[rng.randrange(len(games))] for _ in range(len(games))]
        boot[b] = _calibrate(_fit(samp, len(ids), idx, iters=1500), anchors)
    ci = 1.96 * boot.std(axis=0)

    order = sorted(range(len(ids)), key=lambda i: -cal[i])
    print(f"{'bot':>16} {'Elo':>6} {'±95':>5} {'games':>6}  {'ref':>5}")
    for i in order:
        rf = f"{ref[ids[i]]}" if ref[ids[i]] is not None else "  -"
        print(f"{ids[i]:>16} {cal[i]:>6.0f} {ci[i]:>5.0f} {gcount[ids[i]]:>6}  {rf:>5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
