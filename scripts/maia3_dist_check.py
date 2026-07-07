"""Phase A: extract Maia-3's FULL move distribution (soft labels) and check their quality as a
distillation target. The premise "Maia-3 pre-smoothed the noise" requires its distribution to be
SMOOTH (not near-one-hot) and CALIBRATED (predicted prob ~ empirical accuracy). Reports entropy,
top-1 prob, top-3 mass, effective #moves, mass on the actual human move, top-1 acc, and calibration.
"""
from __future__ import annotations
import argparse, os, sys, math, numpy as np, torch
from collections import deque, defaultdict
import chess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_maia3_vs_ours import sample_positions, BANDS, DEFAULT_PGN
from maia3.uci import parse_args as maia3_parse_args, Maia3UCIEngine
from maia3.dataset import tokenize_board, get_legal_moves_mask

_NEG = float("-inf")


def build_engine(model_name, device):
    cfg = maia3_parse_args(["--model", model_name, "--device", device, "--use-uci-history"])
    eng = Maia3UCIEngine(cfg); eng.ensure_model_loaded()
    return eng


@torch.no_grad()
def maia3_dist(eng, board, move_stack, self_elo, oppo_elo, dev):
    """Full {uci: prob} over legal moves, replicating Maia3UCIEngine.score_moves' masked softmax."""
    eng.board = board
    h = deque(maxlen=eng.cfg.history); b = chess.Board(); h.append(tokenize_board(b))
    for mv in move_stack:
        b.push(mv); h.append(tokenize_board(b))
    eng.history = h; eng.self_elo = int(self_elo); eng.oppo_elo = int(oppo_elo)
    tokens = eng._tokens_from_history(eng.history).unsqueeze(0).to(dev)
    se = torch.tensor([int(self_elo)], dtype=torch.long, device=dev)
    oe = torch.tensor([int(oppo_elo)], dtype=torch.long, device=dev)
    logits_move, _v, _ = eng.model(tokens, se, oe)
    logits = logits_move[0].float()
    mask = get_legal_moves_mask(board, eng.all_moves_dict).to(dev)
    logits = logits.masked_fill(~mask, _NEG)
    probs = torch.softmax(logits, dim=-1)
    out = {}
    for idx in mask.nonzero().flatten().tolist():
        mv = eng._move_from_index(idx)
        if mv is not None:
            out[mv.uci()] = float(probs[idx])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maia3-model", default="maia3-23m")
    ap.add_argument("--pgn", default=DEFAULT_PGN)
    ap.add_argument("--per-band", type=int, default=50)
    ap.add_argument("--min-ply", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); dev = a.device
    samples, counts = sample_positions(a.pgn, a.per_band, a.min_ply, a.seed)
    print(f"got {len(samples)} samples", flush=True)
    eng = build_engine(a.maia3_model, dev)
    print("maia3 loaded", flush=True)

    ent, top1p, top3m, effN, human_p, hit = [], [], [], [], [], []
    n_legal = []
    for i, s in enumerate(samples):
        d = maia3_dist(eng, s["board"], list(s["board"].move_stack), s["self_elo"], s["oppo_elo"], dev)
        if not d: continue
        ps = np.array(sorted(d.values(), reverse=True))
        ps = ps / ps.sum()
        H = float(-(ps * np.log2(ps + 1e-12)).sum())
        ent.append(H); top1p.append(float(ps[0])); top3m.append(float(ps[:3].sum()))
        effN.append(2 ** H); n_legal.append(len(ps))
        human_p.append(float(d.get(s["actual"], 0.0)))
        hit.append(int(max(d, key=d.get) == s["actual"]))
        if (i + 1) % 200 == 0: print(f"  {i+1}/{len(samples)}", flush=True)

    ent = np.array(ent); top1p = np.array(top1p); human_p = np.array(human_p); hit = np.array(hit)
    print(f"\n===== MAIA-3 SOFT-LABEL QUALITY (N={len(ent)}) =====")
    print(f"  entropy (bits):     mean={ent.mean():.2f}  median={np.median(ent):.2f}  (effective #moves mean={np.mean(effN):.1f}, avg legal={np.mean(n_legal):.1f})")
    print(f"  top-1 prob:         mean={top1p.mean():.3f}  median={np.median(top1p):.3f}   top-3 mass mean={np.mean(top3m):.3f}")
    print(f"  P(actual human mv): mean={human_p.mean():.3f}  median={np.median(human_p):.3f}   (0-mass on human: {100*np.mean(human_p<1e-6):.1f}%)")
    print(f"  top-1 acc (argmax==human): {100*hit.mean():.1f}%")
    print("\n  smoothness read: near-one-hot (top-1~1.0, entropy~0) => soft labels ~= one-hot (premise weak);")
    print("                   spread (entropy>1.5, top-1<0.6)      => rich soft target (premise holds)")

    # calibration: bin by top-1 prob, compare to empirical top-1 accuracy
    print("\n  CALIBRATION (top-1 prob bin -> empirical argmax accuracy):")
    edges = [0, 0.3, 0.45, 0.6, 0.75, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (top1p >= lo) & (top1p < hi)
        if m.sum() >= 5:
            print(f"    pred {lo:.2f}-{hi:.2f} (n={int(m.sum()):4d}): mean-pred={top1p[m].mean():.3f}  actual-acc={hit[m].mean():.3f}")


if __name__ == "__main__":
    main()
