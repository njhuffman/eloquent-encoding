"""Position sharpness: for each position, evaluate EVERY legal move (flat eval head) and measure
  - blunder_density = fraction of legal moves with Δeval < -0.2 (how many ways to blunder)
  - eval_spread     = max - min of our-value-after over legal moves (how much move choice matters)
Compare HUMAN positions (from human games) vs the model's SELF-PLAY positions, per elo band.
Tests whether the model reaches QUIETER positions in its own games (fewer blunder opportunities),
which would explain rating above target despite matching per-position blunder rate on human positions."""
from __future__ import annotations
import argparse, sys, os, numpy as np, torch, chess, h5py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_policy.flat_policy import FlatMultiTaskPolicy
from style_policy.board_encode import packed_to_board, board_to_packed
from style_policy.model_spec import elo_to_bucket
from style_policy import move_index

DEV = "cuda"; NEG = -1e9; TH = -0.20
CK = "style_policy_checkpoints/flat_multitask_128M/flat_multitask_128M.pt"
IDXF = torch.tensor(move_index.IDX_FROM); IDXT = torch.tensor(move_index.IDX_TO)


class Model:
    def __init__(self):
        ck = torch.load(CK, map_location=DEV)
        self.m = FlatMultiTaskPolicy.from_config(ck["architecture"]); self.m.load_state_dict(ck["model"], strict=False)
        self.m.to(DEV).eval(); self.n_elo = int(ck["architecture"]["n_elo_buckets"])
        global IDXF, IDXT; IDXF, IDXT = IDXF.to(DEV), IDXT.to(DEV)

    @torch.no_grad()
    def value(self, packed_int64):
        out = []
        for i in range(0, len(packed_int64), 1024):
            pk = torch.from_numpy(packed_int64[i:i+1024]).to(DEV)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                cls, _ = self.m.encode(pk, hist=None); out.append(self.m.eval_value(cls).float().cpu())
        return torch.cat(out).numpy()


def _term_val(b):
    return -1.0 if b.is_checkmate() else 0.0


def sharpness(mo, boards):
    """Return per-board (blunder_density, eval_spread)."""
    V_before = mo.value(np.stack([board_to_packed(b) for b in boards]).astype(np.int64))
    rows = []                                               # (board_idx, resulting_value_after_from_mover)
    flat_pk = []; owner = []; term_val = []
    for bi, b in enumerate(boards):
        for mv in b.legal_moves:
            b.push(mv)
            if b.is_game_over():
                term_val.append(-_term_val(b)); flat_pk.append(None); owner.append(bi)   # our value after = -STM(terminal)
            else:
                term_val.append(None); flat_pk.append(board_to_packed(b).astype(np.int64)); owner.append(bi)
            b.pop()
    real = [i for i, p in enumerate(flat_pk) if p is not None]
    Vr = mo.value(np.stack([flat_pk[i] for i in real]).astype(np.int64)) if real else np.array([])
    val_after = np.zeros(len(flat_pk))                      # our value after each move
    for k, i in enumerate(real):
        val_after[i] = -Vr[k]                               # resulting is opponent-to-move -> negate
    for i, tv in enumerate(term_val):
        if tv is not None:
            val_after[i] = tv
    owner = np.array(owner)
    dens = np.zeros(len(boards)); spread = np.zeros(len(boards))
    for bi in range(len(boards)):
        m = owner == bi
        d = val_after[m] - V_before[bi]                     # Δeval per legal move
        dens[bi] = np.mean(d < TH); spread[bi] = (val_after[m].max() - val_after[m].min()) if m.any() else 0.0
    return dens, spread


def _move(board, i):
    f, t = int(IDXF[i]), int(IDXT[i])
    pr = chess.QUEEN if (board.piece_type_at(f) == chess.PAWN and chess.square_rank(t) in (0, 7)) else None
    mv = chess.Move(f, t, promotion=pr); return mv if mv in board.legal_moves else None


@torch.no_grad()
def selfplay_positions(mo, band, n_games, max_plies, temp, want):
    boards = [chess.Board() for _ in range(n_games)]; fin = [False] * n_games; visited = []
    eidx = elo_to_bucket(torch.tensor([band]), mo.n_elo).to(DEV)
    for ply in range(max_plies):
        live = [i for i in range(n_games) if not fin[i] and not boards[i].is_game_over()]
        for i in range(n_games):
            if not fin[i] and boards[i].is_game_over(): fin[i] = True
        if not live: break
        pk = np.stack([board_to_packed(boards[i]) for i in live]).astype(np.int64)
        msk = torch.from_numpy(np.stack([move_index.legal_index_mask(boards[i]) for i in live])).to(DEV)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            cls, sq = mo.m.encode(torch.from_numpy(pk).to(DEV), hist=None)
            lg = mo.m.human_logits(cls, sq, eidx.expand(len(live))).float().masked_fill(~msk, NEG)
        a = torch.multinomial(torch.softmax(lg / temp, 1), 1).squeeze(1)
        for k, i in enumerate(live):
            if ply >= 6: visited.append(boards[i].copy())   # skip opening moves (all similar)
            mv = _move(boards[i], int(a[k]))
            if mv is None: fin[i] = True; continue
            boards[i].push(mv)
    rng = np.random.default_rng(0)
    return [visited[j] for j in rng.choice(len(visited), min(want, len(visited)), replace=False)]


def human_positions(f, elo, band, w, want):
    pool = np.nonzero(np.abs(elo - band) <= w)[0]
    idx = np.sort(np.random.default_rng(band).choice(pool, want, replace=False))
    return [packed_to_board(p.astype(np.uint8)) for p in f["packed_pre"][idx]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/eloquence_bulk/databases/wdl_history_128M.h5")
    ap.add_argument("--bands", type=int, nargs="+", default=[1000, 1900]); ap.add_argument("--n", type=int, default=700)
    ap.add_argument("--games", type=int, default=40); ap.add_argument("--w", type=int, default=60)
    ap.add_argument("--out", default="/workspaces/eloquent-encoding/position_sharpness.png")
    a = ap.parse_args()
    mo = Model(); f = h5py.File(a.data, "r"); print("loading elo ...", flush=True); elo = f["elo_to_move"][:]
    fig, ax = plt.subplots(1, len(a.bands), figsize=(6 * len(a.bands), 4.5)); ax = np.atleast_1d(ax)
    for c, band in enumerate(a.bands):
        hb = human_positions(f, elo, band, a.w, a.n)
        sb = selfplay_positions(mo, band, a.games, 160, 1.0, a.n)
        hd, hsp = sharpness(mo, hb); sd, ssp = sharpness(mo, sb)
        print(f"\nband {band}:  blunder-density  human {100*hd.mean():.1f}%  self-play {100*sd.mean():.1f}%   "
              f"|  eval-spread  human {hsp.mean():.3f}  self-play {ssp.mean():.3f}", flush=True)
        bins = np.linspace(0, 0.6, 40)
        ax[c].hist(hd, bins=bins, density=True, alpha=0.55, label=f"human pos (mean {100*hd.mean():.0f}%)", color="#2c7fb8")
        ax[c].hist(sd, bins=bins, density=True, alpha=0.55, label=f"self-play pos (mean {100*sd.mean():.0f}%)", color="#e6772e")
        ax[c].set_title(f"band {band}: fraction of legal moves that blunder"); ax[c].set_xlabel("blunder density"); ax[c].legend()
    plt.tight_layout(); plt.savefig(a.out, dpi=110); print(f"\nsaved {a.out}", flush=True)


if __name__ == "__main__":
    main()
