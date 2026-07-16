"""Per-move value-swing distributions: real human moves vs the elo-conditioned head's self-play,
at bands 1000/1500/1900. Δ = (mover's value AFTER the move) - (value BEFORE), from the eval head
(objective, depth-8 SF) and the WDL head. Blunders = large negative Δ. Overlaid histograms ask:
does the head blunder at the same FREQUENCY and SCALE as real humans of that elo?

value convention: eval_value/wdl_value are STM-relative in [-1,1]; after a move the opponent is to
move, so mover's value-after = -value(next_pos). Δ = -value(next) - value(cur). Terminal next_pos:
-1 if checkmate (STM lost), else 0 (draw)."""
from __future__ import annotations
import argparse, sys, os, numpy as np, torch, chess, h5py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_policy.flat_policy import FlatMultiTaskPolicy
from style_policy.board_encode import packed_to_board, board_to_packed
from style_policy.model_spec import elo_to_bucket
from style_policy import move_index

DEV = "cuda"; NEG = -1e9
IDXF = torch.tensor(move_index.IDX_FROM); IDXT = torch.tensor(move_index.IDX_TO)


def _move(board, i):
    f, t = int(IDXF[i]), int(IDXT[i])
    pr = chess.QUEEN if (board.piece_type_at(f) == chess.PAWN and chess.square_rank(t) in (0, 7)) else None
    mv = chess.Move(f, t, promotion=pr)
    return mv if mv in board.legal_moves else None


class M:
    def __init__(self, ckpt):
        ck = torch.load(ckpt, map_location=DEV)
        self.m = FlatMultiTaskPolicy.from_config(ck["architecture"]); self.m.load_state_dict(ck["model"], strict=False)
        self.m.to(DEV).eval(); self.n_elo = int(ck["architecture"]["n_elo_buckets"])
        global IDXF, IDXT; IDXF, IDXT = IDXF.to(DEV), IDXT.to(DEV)

    @torch.no_grad()
    def values(self, packed_int64, band):                       # -> (eval_value, wdl_value) STM, per position
        ev, wv = [], []
        for i in range(0, len(packed_int64), 512):
            pk = torch.from_numpy(packed_int64[i:i+512]).to(DEV)
            eidx = elo_to_bucket(torch.full((len(pk),), band), self.n_elo).to(DEV)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                cls, _ = self.m.encode(pk, hist=None)
                ev.append(self.m.eval_value(cls).float().cpu())
                w = torch.softmax(self.m.value_head(cls, elo_idx=eidx).float(), -1)
                wv.append((w[:, 2] - w[:, 0]).cpu())
        return torch.cat(ev).numpy(), torch.cat(wv).numpy()


def _term_val(board):                                           # STM value of a terminal position
    return -1.0 if board.is_checkmate() else 0.0


def human_deltas(mo, data, elo_all, band, n, w=60):
    pool = np.nonzero(np.abs(elo_all - band) <= w)[0]
    idx = np.sort(np.random.default_rng(band).choice(pool, min(n, len(pool)), replace=False))
    packed = data["packed_pre"][idx]; hf = data["from_sq"][idx]; ht = data["to_sq"][idx]
    cur = packed.astype(np.int64)
    nxt = cur.copy(); term_e = np.zeros(len(idx)); term_w = np.zeros(len(idx)); is_term = np.zeros(len(idx), bool)
    keep = np.ones(len(idx), bool)
    for j in range(len(idx)):
        b = packed_to_board(packed[j].astype(np.uint8))
        try:
            mv = b.find_move(int(hf[j]), int(ht[j]))
        except Exception:
            keep[j] = False; continue
        b.push(mv)
        if b.is_game_over():
            is_term[j] = True; term_e[j] = _term_val(b); term_w[j] = _term_val(b)
        else:
            nxt[j] = board_to_packed(b).astype(np.int64)
    Vc_e, Vc_w = mo.values(cur, band); Vn_e, Vn_w = mo.values(nxt, band)
    Vn_e = np.where(is_term, term_e, Vn_e); Vn_w = np.where(is_term, term_w, Vn_w)
    de = (-Vn_e - Vc_e)[keep]; dw = (-Vn_w - Vc_w)[keep]
    return de, dw


@torch.no_grad()
def selfplay_deltas(mo, band, n_games, max_plies, temp):
    boards = [chess.Board() for _ in range(n_games)]
    eidx = elo_to_bucket(torch.tensor([band]), mo.n_elo).to(DEV)
    seq_e = [[] for _ in range(n_games)]; seq_w = [[] for _ in range(n_games)]
    fin = [None] * n_games                                       # terminal STM value once game ends
    for ply in range(max_plies):
        for i in range(n_games):
            if fin[i] is None and boards[i].is_game_over():
                fin[i] = _term_val(boards[i])
        live = [i for i in range(n_games) if fin[i] is None]
        if not live:
            break
        packed = np.stack([board_to_packed(boards[i]) for i in live]).astype(np.int64)
        masks = torch.from_numpy(np.stack([move_index.legal_index_mask(boards[i]) for i in live])).to(DEV)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            cls, sq = mo.m.encode(torch.from_numpy(packed).to(DEV), hist=None)
            ev = mo.m.eval_value(cls).float().cpu().numpy()
            w = torch.softmax(mo.m.value_head(cls, elo_idx=eidx.expand(len(live))).float(), -1)
            wv = (w[:, 2] - w[:, 0]).cpu().numpy()
            logits = mo.m.human_logits(cls, sq, eidx.expand(len(live))).float().masked_fill(~masks, NEG)
        a = torch.multinomial(torch.softmax(logits / temp, 1), 1).squeeze(1)
        for k, i in enumerate(live):
            seq_e[i].append(float(ev[k])); seq_w[i].append(float(wv[k]))
            mv = _move(boards[i], int(a[k]))
            if mv is None:
                fin[i] = 0.0; continue
            boards[i].push(mv)
    de, dw = [], []
    for i in range(n_games):
        for t in range(len(seq_e[i])):
            vn_e = seq_e[i][t + 1] if t + 1 < len(seq_e[i]) else fin[i]
            vn_w = seq_w[i][t + 1] if t + 1 < len(seq_w[i]) else fin[i]
            if vn_e is None:
                continue
            de.append(-vn_e - seq_e[i][t]); dw.append(-vn_w - seq_w[i][t])
    return np.array(de), np.array(dw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="style_policy_checkpoints/flat_multitask_128M/flat_multitask_128M.pt")
    ap.add_argument("--data", default="/mnt/eloquence_bulk/databases/wdl_history_128M.h5")
    ap.add_argument("--bands", type=int, nargs="+", default=[1000, 1500, 1900])
    ap.add_argument("--n-human", type=int, default=6000); ap.add_argument("--games", type=int, default=48)
    ap.add_argument("--max-plies", type=int, default=140); ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--out", default="/workspaces/eloquent-encoding/game_deltas.png")
    a = ap.parse_args()
    mo = M(a.ckpt)
    data = h5py.File(a.data, "r"); print("loading elo column ...", flush=True); elo_all = data["elo_to_move"][:]
    R = {}
    for band in a.bands:
        hd_e, hd_w = human_deltas(mo, data, elo_all, band, a.n_human)
        sp_e, sp_w = selfplay_deltas(mo, band, a.games, a.max_plies, a.temp)
        R[band] = dict(he=hd_e, hw=hd_w, se=sp_e, sw=sp_w)
        def bl(x, th=-0.20): return 100 * np.mean(x < th)
        print(f"band {band}: human n={len(hd_e)} selfplay n={len(sp_e)} | "
              f"eval-blunder%(Δ<-0.2) human {bl(hd_e):.1f} vs self {bl(sp_e):.1f} | "
              f"mean Δeval human {hd_e.mean():+.3f} self {sp_e.mean():+.3f}", flush=True)
    fig, ax = plt.subplots(2, len(a.bands), figsize=(4.2 * len(a.bands), 7))
    bins_e = np.linspace(-1.2, 0.6, 60); bins_w = np.linspace(-1.5, 0.8, 60)
    for c, band in enumerate(a.bands):
        for r, (key_h, key_s, bins, lab) in enumerate([("he", "se", bins_e, "Δ eval-head value"),
                                                        ("hw", "sw", bins_w, "Δ WDL value")]):
            ax[r, c].hist(R[band][key_h], bins=bins, density=True, alpha=0.55, label="human", color="#2c7fb8")
            ax[r, c].hist(R[band][key_s], bins=bins, density=True, alpha=0.55, label=f"self-play @{band}", color="#e6772e")
            ax[r, c].set_title(f"band {band}: {lab}"); ax[r, c].set_yscale("log"); ax[r, c].legend(fontsize=8)
            ax[r, c].axvline(0, color="k", lw=0.6, ls=":")
    plt.tight_layout(); plt.savefig(a.out, dpi=110); print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
