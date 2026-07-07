"""3-model human-move panel: ours + Maia-2 + Maia-3, on held-out 2025-05 positions (true human
move = ground truth). Answers: is there DIVERSITY to exploit? i.e. how often is one model wrong
but the other two agree on the human move (real ensemble recovery), vs redundant overlap.

Reports per-model top-1 move-match, agreement structure (# models correct), pairwise agreement,
majority-vote accuracy, oracle (>=1 correct), and per-model recovery (model wrong & other two right).
CPU (our model + Maia-3 UCI); Maia-2 on GPU if available. Slow per-position -> keep per-band modest.
"""
from __future__ import annotations
import argparse, os, sys, torch
from collections import defaultdict, Counter
import chess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_maia3_vs_ours import sample_positions, our_top1, _load_our_model, BANDS, DEFAULT_PGN, DEFAULT_CKPT
from style_policy.maia3_bot import Maia3Bot
from style_policy.maia2_bot import load_maia2

MODELS = ["ours", "maia2", "maia3"]


def maia2_top1(m, prep, inference, board, self_elo, opp_elo):
    move_probs, _ = inference.inference_each(m, prep, board.fen(), self_elo, opp_elo)
    legal = {mv.uci() for mv in board.legal_moves}
    items = [(u, p) for u, p in move_probs.items() if u in legal]
    return max(items, key=lambda x: x[1])[0] if items else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--our-ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--pgn", default=DEFAULT_PGN)
    ap.add_argument("--maia3-model", default="maia3-23m")
    ap.add_argument("--per-band", type=int, default=80)
    ap.add_argument("--min-ply", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(); torch.set_num_threads(a.threads)

    samples, counts = sample_positions(a.pgn, a.per_band, a.min_ply, a.seed)
    print(f"got {len(samples)} samples ({sum(1 for b in BANDS if counts[b] < a.per_band)} bands under-filled)", flush=True)

    model, arch, n_ply = _load_our_model(a.our_ckpt)
    for s in samples:
        s["ours"] = our_top1(model, n_ply, s["board"], s["self_elo"])
    print("scored ours", flush=True)

    from maia2 import inference
    m2, prep = load_maia2("rapid", device="gpu")
    for i, s in enumerate(samples):
        s["maia2"] = maia2_top1(m2, prep, inference, s["board"], s["self_elo"], s["oppo_elo"])
        if (i + 1) % 200 == 0: print(f"  maia2 {i+1}/{len(samples)}", flush=True)
    print("scored maia2", flush=True)

    bot = Maia3Bot(model=a.maia3_model, self_elo=samples[0]["self_elo"], oppo_elo=samples[0]["oppo_elo"])
    try:
        for i, s in enumerate(samples):
            bot.set_elos(s["self_elo"], s["oppo_elo"])
            mv = bot.choose_move(s["board"]); s["maia3"] = mv.uci() if mv is not None else None
            if (i + 1) % 200 == 0: print(f"  maia3 {i+1}/{len(samples)}", flush=True)
    finally:
        bot.close()
    print("scored maia3\n", flush=True)

    n = len(samples)
    def correct(s, m): return s[m] is not None and s[m] == s["actual"]

    print(f"===== 3-MODEL PANEL (N={n}, held-out 2025-05) =====")
    for m in MODELS:
        print(f"  {m:6s} top-1 move-match: {100*sum(correct(s, m) for s in samples)/n:.2f}%")

    kc = Counter(sum(correct(s, m) for m in MODELS) for s in samples)
    print("\n#models correct on a position:")
    for k in range(4):
        print(f"  {k}/3 correct: {kc[k]:5d}  ({100*kc[k]/n:.1f}%)")

    print("\npairwise agreement (same top-1 move, regardless of correctness):")
    for i in range(len(MODELS)):
        for j in range(i+1, len(MODELS)):
            ag = sum(samples[x][MODELS[i]] == samples[x][MODELS[j]] for x in range(n)) / n
            print(f"  {MODELS[i]} vs {MODELS[j]}: {100*ag:.1f}%")

    # majority vote (>=2 agree -> that move; all differ -> fallback to strongest single = maia3)
    maj_hit = cover = 0
    for s in samples:
        votes = Counter(s[m] for m in MODELS if s[m] is not None)
        mv, ct = votes.most_common(1)[0]
        if ct >= 2:
            cover += 1; maj_hit += (mv == s["actual"])
        else:
            maj_hit += correct(s, "maia3")
    best = max(100*sum(correct(s, m) for s in samples)/n for m in MODELS)
    oracle = 100*sum(any(correct(s, m) for m in MODELS) for s in samples)/n
    print(f"\nmajority-vote acc: {100*maj_hit/n:.2f}%  (best single {best:.2f}%, oracle>=1 {oracle:.2f}%)"
          f"  | majority exists on {100*cover/n:.1f}% of positions")

    print("\nRECOVERY — model WRONG but other two BOTH match the human move (real ensemble gain):")
    for m in MODELS:
        others = [o for o in MODELS if o != m]
        rescued = sum((not correct(s, m)) and correct(s, others[0]) and correct(s, others[1]) for s in samples)
        lone = sum(correct(s, m) and (not correct(s, others[0])) and (not correct(s, others[1])) for s in samples)
        print(f"  {m:6s}: rescued-by-other-two {rescued:4d} ({100*rescued/n:.1f}%)   |   "
              f"lone-correct (would be outvoted) {lone:4d} ({100*lone/n:.1f}%)")


if __name__ == "__main__":
    main()
