"""Human-fidelity across 3 vectors for distill vs one-hot (vs flat), on the SAME human positions:
  (1) move-distribution:  human-move CE (perplexity) + top-1 match  -> does the learned distribution
      put mass where humans actually move?
  (2) blunder frequency:  sample the model's move (natural T=1.0), score Δeval via the flat eval head,
      P(Δeval < -0.2), vs the human's actual move on the same position.
  (3) blunder scale:      mean |Δeval| given a blunder.
Hypothesis: distill's soft targets reproduce human error freq+scale at natural T (encoding the
small-mistake mass) where one-hot needs a temperature hack that can't match both.

Handles factored MultiBandPolicy (distill_*) and flat FlatMultiTaskPolicy. Δeval scored by the flat
model's eval head (consistent with game_deltas.py). value convention as in game_deltas."""
from __future__ import annotations
import argparse, sys, os, numpy as np, torch, chess, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_policy.multiband_policy import MultiBandPolicy
from style_policy.flat_policy import FlatMultiTaskPolicy
from style_policy.board_encode import packed_to_board, board_to_packed, legal_from_u64, legal_to_u64
from style_policy.legal_mask import u64_to_mask
from style_policy.model_spec import elo_to_bucket
from style_policy import move_index

DEV = "cuda"; NEG = -1e9; TH = -0.20
FLAT_CK = "style_policy_checkpoints/flat_multitask_128M/flat_multitask_128M.pt"


def _u64mask(u64):  # single int -> (64,) bool  (reinterpret u64 bits as int64)
    v = np.array([int(u64)], dtype=np.uint64).view(np.int64)
    return u64_to_mask(torch.from_numpy(v)).squeeze(0)


class Scorer:                                                    # flat eval head as the objective judge
    def __init__(self):
        ck = torch.load(FLAT_CK, map_location=DEV)
        self.m = FlatMultiTaskPolicy.from_config(ck["architecture"]); self.m.load_state_dict(ck["model"], strict=False)
        self.m.to(DEV).eval()

    @torch.no_grad()
    def value(self, packed_int64):                              # STM eval value per position
        out = []
        for i in range(0, len(packed_int64), 512):
            pk = torch.from_numpy(packed_int64[i:i+512]).to(DEV)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                cls, _ = self.m.encode(pk, hist=None)
                out.append(self.m.eval_value(cls).float().cpu())
        return torch.cat(out).numpy()


def load_any(ckpt):
    ck = torch.load(ckpt, map_location=DEV)
    arch = ck["architecture"]; flat = "nnue_head" in arch or arch.get("d_model", 0) == 512 and "bands" not in ck
    if "sf" in ckpt or "flat" in ckpt:                          # flat model
        m = FlatMultiTaskPolicy.from_config(arch); m.load_state_dict(ck["model"], strict=False); m.to(DEV).eval()
        return m, "flat", int(arch["n_elo_buckets"])
    m = MultiBandPolicy.from_config(arch); m.load_state_dict(ck["model"], strict=False); m.to(DEV).eval()
    return m, "factored", int(arch.get("n_elo_buckets", 0))


@torch.no_grad()
def analyze(name, ckpt, boards, packed, hf, ht, fromleg, toleg, band, scorer, V_before):
    m, kind, n_elo = load_any(ckpt)
    N = len(boards); dev = DEV
    ce_sum = 0.0; top1 = 0; ent_sum = 0.0
    sampled_after = packed.copy(); term = np.zeros(N); is_term = np.zeros(N, bool)
    g = torch.Generator(device=dev).manual_seed(1)
    for i in range(0, N, 256):
        sl = slice(i, i + 256); pk = torch.from_numpy(packed[sl].astype(np.int64)).to(dev)
        b_hf = torch.from_numpy(hf[sl].astype(np.int64)).to(dev); b_ht = torch.from_numpy(ht[sl].astype(np.int64)).to(dev)
        fmk = u64_to_mask(torch.from_numpy(np.array(fromleg[sl], dtype=np.uint64)).to(torch.int64)).to(dev)
        tmk = u64_to_mask(torch.from_numpy(np.array(toleg[sl], dtype=np.uint64)).to(torch.int64)).to(dev)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            cls, sq = m.encode(pk, hist=None)
            if kind == "factored":
                head = m.heads[int(m.head_index(torch.tensor([band])).item())]
                fl = head.from_logits(sq, cls).float().masked_fill(~fmk, NEG)
                tl = head.to_logits(sq, b_hf, cls).float().masked_fill(~tmk, NEG)      # to|human-from (stored mask)
                lpf = torch.log_softmax(fl, 1); lpt = torch.log_softmax(tl, 1)
                ce = -(lpf.gather(1, b_hf[:, None]).squeeze(1) + lpt.gather(1, b_ht[:, None]).squeeze(1))
                t1 = (fl.argmax(1) == b_hf) & (tl.argmax(1) == b_ht)
                ent = -(lpf.exp() * lpf.clamp(min=-30)).sum(1)                         # from-entropy (proxy)
                frm = torch.multinomial(torch.softmax(fl, 1), 1, generator=g).squeeze(1)
            else:
                eidx = elo_to_bucket(torch.full((len(pk),), band), n_elo).to(dev)
                lmask = torch.from_numpy(np.stack([move_index.legal_index_mask(boards[j]) for j in range(i, min(i+256, N))])).to(dev)
                lg = m.human_logits(cls, sq, eidx).float().masked_fill(~lmask, NEG)
                lp = torch.log_softmax(lg, 1)
                htgt = torch.from_numpy(move_index.move_to_index_arr(hf[sl], ht[sl])).to(dev)
                ce = -lp.gather(1, htgt[:, None]).squeeze(1)
                t1 = lg.argmax(1) == htgt
                ent = -(lp.exp() * lp.clamp(min=-30)).sum(1)
        ce_sum += float(ce.sum()); top1 += int(t1.sum()); ent_sum += float(ent.sum())
        # sample the model's move on each board, apply -> resulting packed
        for k, j in enumerate(range(i, min(i + 256, N))):
            b = boards[j]
            if kind == "factored":
                f = int(frm[k])
                tmask_f = _u64mask(legal_to_u64(b, f)).to(dev)
                tl_f = head.to_logits(sq[k:k+1].float(), torch.tensor([f], device=dev), cls[k:k+1].float()).float().masked_fill(~tmask_f, NEG)
                t = int(torch.multinomial(torch.softmax(tl_f, 1), 1, generator=g).squeeze())
            else:
                a = int(torch.multinomial(torch.softmax(lg[k], 0), 1, generator=g))
                f, t = int(move_index.IDX_FROM[a]), int(move_index.IDX_TO[a])
            try:
                mv = b.find_move(f, t)
            except Exception:
                is_term[j] = True; term[j] = 0.0; continue
            b.push(mv)
            if b.is_game_over():
                is_term[j] = True; term[j] = (-1.0 if b.is_checkmate() else 0.0)
            else:
                sampled_after[j] = board_to_packed(b).astype(np.int64)
            b.pop()
    V_after = scorer.value(sampled_after.astype(np.int64))
    V_after = np.where(is_term, term, V_after)
    d = -V_after - V_before
    bl = d < TH
    return dict(ce=ce_sum / N, top1=100 * top1 / N, ent=ent_sum / N,
                freq=100 * bl.mean(), scale=(float(-d[bl].mean()) if bl.any() else 0.0), d=d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/eloquence_bulk/databases/wdl_history_128M.h5")
    ap.add_argument("--band", type=int, default=1500); ap.add_argument("--n", type=int, default=4000); ap.add_argument("--w", type=int, default=60)
    ap.add_argument("--models", nargs="+", default=[
        "distill_maia3:style_policy_checkpoints/multiband_distill_maia3/multiband_distill_maia3.pt",
        "distill_human:style_policy_checkpoints/multiband_distill_human/multiband_distill_human.pt",
        "flat:style_policy_checkpoints/flat_multitask_128M/flat_multitask_128M.pt"])
    a = ap.parse_args()
    f = h5py.File(a.data, "r"); print("loading elo ...", flush=True); elo = f["elo_to_move"][:]
    pool = np.nonzero(np.abs(elo - a.band) <= a.w)[0]
    idx = np.sort(np.random.default_rng(a.band).choice(pool, min(a.n, len(pool)), replace=False))
    packed = f["packed_pre"][idx].astype(np.int64); hf = f["from_sq"][idx]; ht = f["to_sq"][idx]
    fromleg = f["from_legal_u64"][idx]; toleg = f["to_legal_u64"][idx]
    boards = [packed_to_board(p.astype(np.uint8)) for p in packed]
    scorer = Scorer(); V_before = scorer.value(packed)
    # human actual-move deltas
    ha = packed.copy(); hterm = np.zeros(len(idx)); h_is = np.zeros(len(idx), bool)
    for j in range(len(idx)):
        b = boards[j]
        try: mv = b.find_move(int(hf[j]), int(ht[j]))
        except Exception: h_is[j] = True; continue
        b.push(mv)
        if b.is_game_over(): h_is[j] = True; hterm[j] = (-1.0 if b.is_checkmate() else 0.0)
        else: ha[j] = board_to_packed(b).astype(np.int64)
        b.pop()
    Vah = np.where(h_is, hterm, scorer.value(ha)); dh = -Vah - V_before; blh = dh < TH
    print(f"\n===== HUMAN-FIDELITY @band {a.band} (n={len(idx)}) — Δeval by flat eval head =====")
    print(f"{'model':<15}{'human_CE':>9}{'top1%':>7}{'entropy':>8}{'blund_freq%':>12}{'blund_scale':>12}")
    print(f"{'HUMAN(actual)':<15}{'-':>9}{'-':>7}{'-':>8}{100*blh.mean():>11.1f}{float(-dh[blh].mean()):>12.3f}")
    for spec in a.models:
        nm, ckpt = spec.split(":", 1)
        r = analyze(nm, ckpt, boards, packed, hf, ht, fromleg, toleg, a.band, scorer, V_before)
        print(f"{nm:<15}{r['ce']:>9.3f}{r['top1']:>7.1f}{r['ent']:>8.2f}{r['freq']:>11.1f}{r['scale']:>12.3f}", flush=True)


if __name__ == "__main__":
    main()
