"""Compare the SF-eval multi-task model vs its no-eval baseline on held-out val positions:
  (1) WDL head:  CE + 3-way accuracy vs game result.
  (2) SF-eval:   held-out depth-1 cp decodability -> linear-probe Pearson r on FROZEN CLS features
                 for both encoders (does co-training make eval more linearly present?), plus the
                 multi-task model's zero-shot nnue_head r/MSE (its actually-trained head).
Positions are freshly labeled with depth-1 SF eval so nothing is train-contaminated."""
from __future__ import annotations
import argparse, sys, os, numpy as np, torch, h5py, chess, chess.engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_policy.multiband_policy import MultiBandPolicy
from style_policy.model_spec import elo_to_bucket
from style_policy.loss import wdl_ce
from style_policy.board_encode import packed_to_board
from dataset_generation.stockfish_eval import CP_CLAMP

def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location=device)
    m = MultiBandPolicy.from_config(ck["architecture"]); m.load_state_dict(ck["model"], strict=False)
    return m.to(device).eval(), ck["architecture"]

@torch.no_grad()
def encode_all(model, packed, device, bs=256):
    cls = []
    for i in range(0, len(packed), bs):
        p = torch.from_numpy(packed[i:i+bs].astype(np.uint8)).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            c, _ = model.encode(p, hist=None)
        cls.append(c.float().cpu())
    return torch.cat(cls, 0)

@torch.no_grad()
def wdl_metrics(model, cls, elo, result, n_elo, device):
    ce = 0.0; correct = 0; n = 0
    for i in range(0, len(cls), 4096):
        c = cls[i:i+4096].to(device); e = elo[i:i+4096]; r = result[i:i+4096].to(device)
        logits = model.value_head(c, elo_idx=elo_to_bucket(e, n_elo).to(device))
        ce += float(wdl_ce(logits, r)) * len(c)
        correct += int((logits.argmax(-1) == r).sum()); n += len(c)
    return ce / n, 100.0 * correct / n

def probe_r(feat_tr, y_tr, feat_te, y_te, alpha=10.0):
    """Ridge linear probe (closed form) on standardized features -> held-out Pearson r."""
    mu = feat_tr.mean(0, keepdim=True); sd = feat_tr.std(0, keepdim=True) + 1e-6
    Xtr = (feat_tr - mu) / sd; Xte = (feat_te - mu) / sd
    Xtr = torch.cat([Xtr, torch.ones(len(Xtr), 1)], 1); Xte = torch.cat([Xte, torch.ones(len(Xte), 1)], 1)
    d = Xtr.shape[1]; A = Xtr.T @ Xtr + alpha * torch.eye(d); w = torch.linalg.solve(A, Xtr.T @ y_tr)
    pred = Xte @ w
    return float(torch.corrcoef(torch.stack([pred, y_te]))[0, 1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="style_policy_checkpoints/multiband_ourdistill_human/multiband_ourdistill_human.pt")
    ap.add_argument("--nnue", default="style_policy_checkpoints/multiband_ourdistill_human_nnue/multiband_ourdistill_human_nnue.pt")
    ap.add_argument("--val", default="/mnt/eloquence_bulk/databases/wdl_validation_2025_05.h5")
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--offset", type=int, default=600000)   # held-out slice, disjoint from training val_sample
    ap.add_argument("--stockfish", default="/usr/games/stockfish")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device if torch.cuda.is_available() else "cpu"

    with h5py.File(a.val, "r") as f:
        sl = slice(a.offset, a.offset + a.n)
        packed = f["packed_pre"][sl]; elo = torch.from_numpy(f["elo_to_move"][sl].astype(np.int64))
        result = torch.from_numpy(f["result"][sl].astype(np.int64))

    print(f"labeling {a.n:,} held-out positions with depth-1 SF eval ...", flush=True)
    eng = chess.engine.SimpleEngine.popen_uci(a.stockfish); eng.configure({"Threads": 1})
    cp = np.empty(a.n, dtype=np.float32)
    for i in range(a.n):
        b = packed_to_board(packed[i].astype(np.uint8))
        cp[i] = eng.analyse(b, chess.engine.Limit(depth=1))["score"].pov(b.turn).score(mate_score=CP_CLAMP)
    eng.quit()
    y = torch.tanh(torch.from_numpy(cp) / 400.0)               # regression target, same as training
    ntr = int(a.n * 0.75); ytr, yte = y[:ntr], y[ntr:]

    rows = []
    for tag, ckpt in [("baseline (no eval)", a.baseline), ("nnue multi-task", a.nnue)]:
        m, arch = load_model(ckpt, dev); n_elo = int(arch["n_elo_buckets"])
        cls = encode_all(m, packed, dev)
        wce, wacc = wdl_metrics(m, cls, elo, result, n_elo, dev)
        r_probe = probe_r(cls[:ntr], ytr, cls[ntr:], yte)
        r_head = mse_head = None
        if getattr(m, "nnue_head", None) is not None:
            with torch.no_grad():
                pred = torch.tanh(m.nnue_head(cls.to(dev)).squeeze(-1)).float().cpu()
            r_head = float(torch.corrcoef(torch.stack([pred[ntr:], yte]))[0, 1])
            mse_head = float(((pred - y) ** 2).mean())
        rows.append((tag, wce, wacc, r_probe, r_head, mse_head))

    print(f"\n===== NNUE MULTI-TASK vs BASELINE (n={a.n:,} held-out, depth-1 eval target) =====")
    print(f"{'model':<22}{'WDL_CE':>8}{'WDL_acc':>9}{'eval_r(probe)':>15}{'eval_r(head)':>14}{'head_mse':>10}")
    for tag, wce, wacc, rp, rh, mh in rows:
        hs = f"{rh:>14.3f}" if rh is not None else f"{'-':>14}"
        ms = f"{mh:>10.4f}" if mh is not None else f"{'-':>10}"
        print(f"{tag:<22}{wce:>8.4f}{wacc:>8.1f}%{rp:>15.3f}{hs}{ms}")
    print("\neval_r(probe): held-out Pearson r of a ridge probe on FROZEN CLS features -> tanh(cp/400).")
    print("eval_r(head):  multi-task model's trained nnue_head, zero-shot. Frozen-encoder probe ceiling was ~0.89.")

if __name__ == "__main__":
    main()
