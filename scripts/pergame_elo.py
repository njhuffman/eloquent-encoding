"""Per-GAME elo aggregate: turn the weak per-position strength signal into a strong per-player
estimate. Train a band classifier on 128M train data, then on held-out 2025-05 games aggregate
the classifier's per-position evidence over each (game, player-color) unit and estimate that
player's rating. Reports per-position vs per-unit elo MAE + accuracy, and MAE vs #positions.
"""
from __future__ import annotations
import argparse, io, sys, numpy as np, torch, h5py, chess.pgn, zstandard
from collections import defaultdict
from style_policy.multiband_policy import MultiBandPolicy
from style_policy.board_encode import board_to_packed
from dataset_generation.candidate_collect import collect_candidate_positions, board_at_ply

BANDS = list(range(1000, 2200, 100))
def bidx(elo): return min(11, max(0, int(elo) // 100 - 10))
def bcenter(b): return 1000 + 100 * b + 50  # band-center elo


def load_model(ckpt, dev):
    ck = torch.load(ckpt, map_location=dev); arch = ck["architecture"]
    m = MultiBandPolicy.from_config(arch).to(dev).eval(); m.load_state_dict(ck["model"])
    n_ply = int(arch.get("n_history_ply", 0)) if arch.get("use_last_move") else 0
    return m, n_ply


@torch.no_grad()
def feats(model, packed, hist, dev, bs=256):
    C, S = [], []
    for i in range(0, len(packed), bs):
        pk = torch.from_numpy(np.asarray(packed[i:i+bs]).astype(np.int64)).to(dev)
        h = tuple(x[i:i+bs].to(dev) for x in hist) if hist is not None else None
        c, s = model.encode(pk, hist=h); C.append(c.float().cpu()); S.append(s.float().mean(1).cpu())
    return torch.cat([torch.cat(C), torch.cat(S)], 1)


def hist_arrays(hist_rows, n_ply):
    hf = np.array([[h[i][0] for i in range(n_ply)] for h in hist_rows], np.int64)
    ht = np.array([[h[i][1] for i in range(n_ply)] for h in hist_rows], np.int64)
    hc = np.array([[h[i][2] for i in range(n_ply)] for h in hist_rows], np.int64)
    return (torch.from_numpy(hf), torch.from_numpy(ht), torch.from_numpy(hc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="style_policy_checkpoints/multiband_history_128M_big/multiband_history_128M_big.pt")
    ap.add_argument("--train-data", default="/mnt/eloquence_bulk/databases/wdl_history_128M.h5")
    ap.add_argument("--pgn", default="/mnt/eloquence_bulk/databases/lichess_db_standard_rated_2025-05_tc_600_0.pgn.zst")
    ap.add_argument("--per-band-train", type=int, default=12000)
    ap.add_argument("--units-per-band", type=int, default=250)
    ap.add_argument("--max-pos-per-unit", type=int, default=24)
    ap.add_argument("--min-pos-per-unit", type=int, default=6)
    ap.add_argument("--max-scan", type=int, default=200000)
    ap.add_argument("--skip-opening", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); dev = a.device; rng = np.random.default_rng(a.seed)
    model, n_ply = load_model(a.ckpt, dev)

    # ---- train band classifier on 128M train (cls+meansq -> 12 bands) ----
    f = h5py.File(a.train_data, "r"); elo = f["elo_to_move"][:]; band = np.clip((elo//100)*100, 1000, 2100)
    sel = np.sort(np.concatenate([rng.choice(np.where(band == b)[0], min(a.per_band_train, int((band == b).sum())), replace=False) for b in BANDS]))
    y = np.array([bidx(elo[i]) for i in sel])
    hist = None
    if n_ply:
        hist = (torch.from_numpy(f["hist_from"][sel][:, :n_ply].astype(np.int64)),
                torch.from_numpy(f["hist_to"][sel][:, :n_ply].astype(np.int64)),
                torch.from_numpy(f["hist_cap"][sel][:, :n_ply].astype(np.int64)))
    print(f"[train] encoding {len(sel):,} positions ...", flush=True)
    X = feats(model, f["packed_pre"][sel], hist, dev); Yt = torch.from_numpy(y)
    clf = torch.nn.Sequential(torch.nn.Linear(768, 256), torch.nn.ReLU(), torch.nn.Linear(256, 12)).to(dev)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(30):
        p = torch.randperm(len(X))
        for i in range(0, len(X), 4096):
            b = p[i:i+4096]
            loss = torch.nn.functional.cross_entropy(clf(X[b].to(dev)), Yt[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
    clf.eval(); del X

    # ---- extract game-grouped positions from held-out 2025-05, balanced by band ----
    print(f"[extract] streaming {a.pgn.split('/')[-1]} ...", flush=True)
    need = {b: a.units_per_band for b in range(12)}
    upacked, uhist, uelo, ubandtrue, uid_of_pos = [], [], [], [], []
    n_units = 0; scanned = 0
    dctx = zstandard.ZstdDecompressor(max_window_size=2**31)
    with open(a.pgn, "rb") as fh:
        text = io.TextIOWrapper(dctx.stream_reader(fh), encoding="utf-8", errors="ignore")
        while scanned < a.max_scan and any(v > 0 for v in need.values()):
            game = chess.pgn.read_game(text)
            if game is None: break
            scanned += 1
            if scanned % 20000 == 0:
                print(f"  scanned {scanned:,} | units {n_units} | remaining {sum(need.values())}", flush=True)
            try:
                mainline, rows = collect_candidate_positions(game, skip_opening_plies=a.skip_opening, exclude_single_legal_move=True)
            except Exception:
                continue
            if not rows: continue
            by_side = defaultdict(list)
            for (ply, stm, elo_tm, opp, res, mv, h) in rows:
                by_side[(stm, elo_tm)].append((ply, h))
            for (stm, elo_tm), plist in by_side.items():
                b = bidx(elo_tm)
                if need[b] <= 0 or len(plist) < a.min_pos_per_unit: continue
                plist = plist[:a.max_pos_per_unit]
                for (ply, h) in plist:
                    board = board_at_ply(mainline, ply)
                    upacked.append(board_to_packed(board)); uhist.append(h)
                    uelo.append(elo_tm); ubandtrue.append(b); uid_of_pos.append(n_units)
                need[b] -= 1; n_units += 1
    print(f"[extract] {n_units} units, {len(upacked):,} positions (scanned {scanned:,})", flush=True)
    print(f"  per-band units: {[a.units_per_band-need[b] for b in range(12)]}")

    # ---- encode extracted positions, per-position classifier probs ----
    hist_e = hist_arrays(uhist, n_ply) if n_ply else None
    Xe = feats(model, upacked, hist_e, dev)
    with torch.no_grad():
        probs = torch.softmax(clf(Xe.to(dev)), 1).cpu().numpy()  # (P,12)
    uid_of_pos = np.array(uid_of_pos); uelo = np.array(uelo); ubandtrue = np.array(ubandtrue)
    band_c = np.array([bcenter(b) for b in range(12)])

    # per-position estimate
    pp_pred_band = probs.argmax(1)
    pp_exp_elo = probs @ band_c
    pp_acc = (pp_pred_band == ubandtrue).mean()
    pp_mae = np.abs(pp_exp_elo - uelo).mean()
    pp_r = np.corrcoef(pp_exp_elo, uelo)[0, 1]
    print(f"\n[per-position] band acc={100*pp_acc:.1f}%  elo MAE={pp_mae:.0f}  Pearson r={pp_r:.3f}")

    # per-unit aggregate (soft: sum log-probs -> argmax band; expected-elo averaged)
    logp = np.log(probs + 1e-9)
    n_pos = np.zeros(n_units); u_true_elo = np.zeros(n_units); u_true_band = np.zeros(n_units, int)
    u_logp = np.zeros((n_units, 12)); u_exp_elo = np.zeros(n_units)
    for i in range(len(uid_of_pos)):
        u = uid_of_pos[i]; u_logp[u] += logp[i]; u_exp_elo[u] += pp_exp_elo[i]; n_pos[u] += 1
        u_true_elo[u] = uelo[i]; u_true_band[u] = ubandtrue[i]
    u_exp_elo /= np.maximum(n_pos, 1)
    u_pred_band = u_logp.argmax(1)
    u_acc = (u_pred_band == u_true_band).mean()
    u_w1 = (np.abs(u_pred_band - u_true_band) <= 1).mean()
    u_mae = np.abs(u_exp_elo - u_true_elo).mean()
    argmax_elo_mae = np.abs(np.array([bcenter(b) for b in u_pred_band]) - u_true_elo).mean()
    base_mae = np.abs(u_true_elo.mean() - u_true_elo).mean()  # predict global mean elo
    r = np.corrcoef(u_exp_elo, u_true_elo)[0, 1]
    # calibrated MAE: undo shrinkage via least-squares fit pred->true (in-sample, illustrative)
    A = np.vstack([u_exp_elo, np.ones_like(u_exp_elo)]).T
    slope, icpt = np.linalg.lstsq(A, u_true_elo, rcond=None)[0]
    cal_mae = np.abs((slope * u_exp_elo + icpt) - u_true_elo).mean()
    print(f"[per-unit ({n_units} game-players, ~{int(n_pos.mean())} pos each)] "
          f"band acc={100*u_acc:.1f}%  within-1={100*u_w1:.1f}%")
    print(f"  elo MAE: expected-elo={u_mae:.0f}  argmax-band={argmax_elo_mae:.0f}  "
          f"calibrated={cal_mae:.0f}  CONSTANT-mean baseline={base_mae:.0f}")
    print(f"  Pearson r(pred,true): per-position={pp_r:.3f} -> per-game={r:.3f}  (aggregation gain)")

    # MAE vs #positions per unit
    print("[MAE vs positions/unit]")
    for lo, hi in [(6, 9), (10, 14), (15, 19), (20, 24)]:
        m = (n_pos >= lo) & (n_pos <= hi)
        if m.sum(): print(f"  {lo}-{hi} pos (n={int(m.sum()):4d}): elo MAE={np.abs(u_exp_elo[m]-u_true_elo[m]).mean():.0f}")


if __name__ == "__main__":
    main()
