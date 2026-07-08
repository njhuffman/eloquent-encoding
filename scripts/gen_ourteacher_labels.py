"""Distillation labels from OUR 128M-big teacher (big->small self-consistent distillation).
Sample positions from wdl_history_128M, run the teacher's per-band factored heads, store the
factored soft targets maia_from = P(from) and maia_to = P(to|true-from) alongside the copied
input fields. Drop-in for the --distill trainer (same schema as gen_maia3_labels). Fast: batched
GPU teacher forward, no PGN/UCI.
"""
from __future__ import annotations
import argparse, numpy as np, torch, h5py
from collections import defaultdict
from style_policy.multiband_policy import MultiBandPolicy
from style_policy.legal_mask import u64_to_mask
from style_policy.packed_codec import PACKED_BOARD_LEN
_NEG = float("-inf")

COPY = ["packed_pre", "from_sq", "to_sq", "promotion", "from_legal_u64", "to_legal_u64",
        "elo_to_move", "opp_elo", "result", "hist_from", "hist_to", "hist_cap"]


def load_teacher(ckpt, dev):
    ck = torch.load(ckpt, map_location=dev); arch = ck["architecture"]
    m = MultiBandPolicy.from_config(arch).to(dev).eval(); m.load_state_dict(ck["model"])
    n_ply = int(arch.get("n_history_ply", 0)) if arch.get("use_last_move") else 0
    return m, n_ply


class Writer:
    def __init__(self, path, src, chunk=8192):
        self.f = h5py.File(path, "w"); self.buf = defaultdict(list)
        self.dt = {k: src[k].dtype for k in COPY}
        self.sh = {k: src[k].shape[1:] for k in COPY}
        for k in COPY:
            self.f.create_dataset(k, (0, *self.sh[k]), maxshape=(None, *self.sh[k]), dtype=self.dt[k], chunks=(chunk, *self.sh[k]))
        for k in ("maia_from", "maia_to"):
            self.f.create_dataset(k, (0, 64), maxshape=(None, 64), dtype=np.float16, chunks=(chunk, 64))

    def add_block(self, cols, mf, mt):
        n = len(mf)
        for k in COPY:
            d = self.f[k]; d.resize(d.shape[0]+n, axis=0); d[-n:] = cols[k]
        for k, v in (("maia_from", mf), ("maia_to", mt)):
            d = self.f[k]; d.resize(d.shape[0]+n, axis=0); d[-n:] = v.astype(np.float16)

    def close(self): self.f.close()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/eloquence_bulk/databases/wdl_history_128M.h5")
    ap.add_argument("--ckpt", default="style_policy_checkpoints/multiband_history_128M_big/multiband_history_128M_big.pt")
    ap.add_argument("--out", default="/mnt/eloquence_bulk/databases/ourteacher_distill_labels.h5")
    ap.add_argument("--n", type=int, default=16_000_000)
    ap.add_argument("--batch", type=int, default=1024); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--teacher-no-hist", action="store_true",
                    help="feed the teacher hist=None -> board-only soft targets (for a no-history student)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); dev = a.device
    teacher, n_ply = load_teacher(a.ckpt, dev)
    if a.teacher_no_hist:
        n_ply = 0  # teacher judges board-only
    src = h5py.File(a.src, "r"); N = src["packed_pre"].shape[0]
    idx = np.sort(np.random.default_rng(a.seed).choice(N, min(a.n, N), replace=False))
    W = Writer(a.out, src)
    print(f"labeling {len(idx):,} positions with teacher {a.ckpt.split('/')[-1]} (n_ply={n_ply}) ...", flush=True)
    done = 0
    for i in range(0, len(idx), a.batch):
        bi = idx[i:i+a.batch]
        cols = {k: src[k][bi] for k in COPY}
        packed = torch.from_numpy(cols["packed_pre"].astype(np.int64)).to(dev)
        elo = torch.from_numpy(cols["elo_to_move"].astype(np.int64))
        fsq = torch.from_numpy(cols["from_sq"].astype(np.int64)).to(dev)
        fmask = u64_to_mask(torch.from_numpy(cols["from_legal_u64"].astype(np.uint64).astype(np.int64)).to(dev))
        tmask = u64_to_mask(torch.from_numpy(cols["to_legal_u64"].astype(np.uint64).astype(np.int64)).to(dev))
        hist = None
        if n_ply:
            hist = (torch.from_numpy(cols["hist_from"][:, :n_ply].astype(np.int64)).to(dev),
                    torch.from_numpy(cols["hist_to"][:, :n_ply].astype(np.int64)).to(dev),
                    torch.from_numpy(cols["hist_cap"][:, :n_ply].astype(np.int64)).to(dev))
        cls, squares = teacher.encode(packed, hist=hist)
        hidx = teacher.head_index(elo).to(dev)
        B = len(bi); fl = squares.new_zeros(B, 64); tl = squares.new_zeros(B, 64)
        for g in range(teacher.n_bands):
            m = hidx == g
            if not bool(m.any()): continue
            fl[m] = teacher.heads[g].from_logits(squares[m], cls[m]).float()
            tl[m] = teacher.heads[g].to_logits(squares[m], fsq[m], cls[m]).float()
        mf = torch.softmax(fl.masked_fill(~fmask, _NEG), -1).cpu().numpy()
        mt = torch.softmax(tl.masked_fill(~tmask, _NEG), -1).cpu().numpy()
        W.add_block(cols, np.nan_to_num(mf), np.nan_to_num(mt))
        done += B
        if done % 500000 < a.batch: print(f"  {done:,}/{len(idx):,}", flush=True)
    W.close(); print(f"DONE: {done:,} rows -> {a.out}")


if __name__ == "__main__":
    main()
