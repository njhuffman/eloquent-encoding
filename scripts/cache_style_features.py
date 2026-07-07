#!/usr/bin/env python3
"""Cache frozen big-rapid-encoder features for a subset of the style-clustering positions.

Encodes a deterministic subset of the style_1500 positions with the FROZEN
MultiBandPolicy encoder (multiband_history_128M_big) and caches the per-position
features -- CLS token (d,) + the 64 square tokens (64,d) -- to an HDF5 file, together
with the move label / legal-mask columns and unit_id / split. The next task (the EM
clustering loop) can then train + eval small heads on these cached features without
re-running the (expensive) encoder.

Subset selection (deterministic):
  * the top --n-nonnovel non-novel units, ranked by n_games desc (tie-break unit_id asc),
  * plus ALL novel units,
  * capped per unit at --max-train train rows and --max-test test rows,
    taking the first-N rows of each split in source row order.

The output h5 is written incrementally with resizable, chunked datasets so we never
hold the full (~50 GB) feature array in RAM. An in-script alignment self-check
re-encodes a handful of source rows and asserts the cached features match, proving the
cached rows line up with their labels.

Mirrors the model-load + history-tensor patterns in scripts/probe_pointer_head.py.
"""
from __future__ import annotations

import argparse
import time

import h5py
import numpy as np
import torch

from style_policy.multiband_policy import MultiBandPolicy

# Columns copied through verbatim (the EM heads need the move label + legal masks).
PASS_COLS = ("from_sq", "to_sq", "from_legal_u64", "to_legal_u64")
PASS_DTYPES = {
    "from_sq": np.uint8,
    "to_sq": np.uint8,
    "from_legal_u64": np.uint64,
    "to_legal_u64": np.uint64,
}


def load_frozen_model(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    arch = ck["architecture"]
    model = MultiBandPolicy.from_config(arch)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, arch


def n_history_ply(arch: dict) -> int:
    return int(arch.get("n_history_ply", 0)) if arch.get("use_last_move") else 0


def select_units(units, n_nonnovel):
    """Return (selected_unit_ids sorted, novel_unit_ids) deterministically."""
    is_novel = units["is_novel"].astype(bool)
    uid = units["unit_id"].astype(np.int64)
    ngames = units["n_games"].astype(np.int64)

    novel_ids = uid[is_novel]
    nn_mask = ~is_novel
    nn_uid = uid[nn_mask]
    nn_games = ngames[nn_mask]
    # rank non-novel by n_games desc, tie-break unit_id asc (fully deterministic)
    order = np.lexsort((nn_uid, -nn_games))
    sel_nonnovel = nn_uid[order][:n_nonnovel]

    sel = np.unique(np.concatenate([sel_nonnovel, novel_ids]))
    return sel, novel_ids


@torch.no_grad()
def encode_batch(model, packed_np, hf_np, ht_np, hc_np, n_ply, device, use_amp):
    """packed_np (B,34) uint8; hist arrays (B,n_ply) int; -> (cls f16, squares f16) numpy."""
    if n_ply > 0:
        hist = (
            torch.from_numpy(hf_np.astype(np.int64)).to(device),
            torch.from_numpy(ht_np.astype(np.int64)).to(device),
            torch.from_numpy(hc_np.astype(np.int64)).to(device),
        )
    else:
        hist = None
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        cls, squares = model.encode(packed_np, hist=hist)
    return cls.float().half().cpu().numpy(), squares.float().half().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/eloquence_bulk/databases/style_1500_2025_01.h5")
    ap.add_argument("--units", default="/mnt/eloquence_bulk/databases/style_1500_2025_01_units.npz")
    ap.add_argument("--ckpt", default="style_policy_checkpoints/multiband_history_128M_big/"
                                      "multiband_history_128M_big.pt")
    ap.add_argument("--out", default="/mnt/eloquence_bulk/databases/style_feat_cache.h5")
    ap.add_argument("--units-out", default="/mnt/eloquence_bulk/databases/style_feat_cache_units.npz")
    ap.add_argument("--n-nonnovel", type=int, default=2000)
    ap.add_argument("--max-train", type=int, default=300)
    ap.add_argument("--max-test", type=int, default=150)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--chunk-rows", type=int, default=200_000)
    ap.add_argument("--h5-chunk", type=int, default=256, help="output h5 chunk rows")
    ap.add_argument("--n-check", type=int, default=8, help="rows to re-encode for the alignment self-check")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    device = a.device
    if device == "cuda" and not torch.cuda.is_available():
        print("cuda not available -> falling back to cpu")
        device = "cpu"
    use_amp = device == "cuda"

    print(f"loading frozen encoder {a.ckpt} on {device} ...")
    model, arch = load_frozen_model(a.ckpt, device)
    d = int(arch["d_model"])
    n_ply = n_history_ply(arch)
    print(f"  d_model={d} n_history_ply={n_ply} use_last_move={arch.get('use_last_move')}")

    units = np.load(a.units, allow_pickle=True)
    sel_ids, novel_ids = select_units(units, a.n_nonnovel)
    sel_set = set(int(x) for x in sel_ids)
    novel_set = set(int(x) for x in novel_ids)
    caps = {0: a.max_train, 1: a.max_test}
    print(f"selected {len(sel_ids)} units ({len(sel_ids) - len(novel_ids)} non-novel + "
          f"{len(novel_ids)} novel); caps train={a.max_train} test={a.max_test}")

    # per-(unit,split) accepted-count tracker
    counts: dict[tuple[int, int], int] = {}

    # rows recorded for the alignment self-check: (cache_idx, src_global_idx)
    check_pairs: list[tuple[int, int]] = []

    src = h5py.File(a.src, "r")
    n_rows = src["packed_pre"].shape[0]

    out = h5py.File(a.out, "w")
    dsets = {}
    dsets["cls"] = out.create_dataset("cls", shape=(0, d), maxshape=(None, d),
                                      dtype=np.float16, chunks=(a.h5_chunk, d))
    dsets["squares"] = out.create_dataset("squares", shape=(0, 64, d), maxshape=(None, 64, d),
                                          dtype=np.float16, chunks=(a.h5_chunk, 64, d))
    dsets["unit_id"] = out.create_dataset("unit_id", shape=(0,), maxshape=(None,),
                                          dtype=np.int32, chunks=(a.h5_chunk,))
    dsets["split"] = out.create_dataset("split", shape=(0,), maxshape=(None,),
                                        dtype=np.int8, chunks=(a.h5_chunk,))
    dsets["src_row"] = out.create_dataset("src_row", shape=(0,), maxshape=(None,),
                                          dtype=np.int64, chunks=(a.h5_chunk,))
    for c in PASS_COLS:
        dsets[c] = out.create_dataset(c, shape=(0,), maxshape=(None,),
                                      dtype=PASS_DTYPES[c], chunks=(a.h5_chunk,))

    n_written = 0

    def append(cls_np, sq_np, buf):
        nonlocal n_written
        b = cls_np.shape[0]
        dsets["cls"].resize(n_written + b, axis=0)
        dsets["cls"][n_written:n_written + b] = cls_np
        dsets["squares"].resize(n_written + b, axis=0)
        dsets["squares"][n_written:n_written + b] = sq_np
        dsets["unit_id"].resize(n_written + b, axis=0)
        dsets["unit_id"][n_written:n_written + b] = buf["unit_id"]
        dsets["split"].resize(n_written + b, axis=0)
        dsets["split"][n_written:n_written + b] = buf["split"]
        dsets["src_row"].resize(n_written + b, axis=0)
        dsets["src_row"][n_written:n_written + b] = buf["src_row"]
        for c in PASS_COLS:
            dsets[c].resize(n_written + b, axis=0)
            dsets[c][n_written:n_written + b] = buf[c]
        n_written += b

    # pending buffers (accepted-but-not-yet-encoded rows), FIFO
    pend = {k: [] for k in ("packed_pre", "hist_from", "hist_to", "hist_cap",
                            "unit_id", "split", "src_row", *PASS_COLS)}
    pend_n = 0

    def flush(force=False):
        nonlocal pend, pend_n
        if pend_n == 0:
            return
        arrs = {k: np.concatenate(v, axis=0) for k, v in pend.items()}
        start = 0
        while pend_n - start >= a.batch or (force and pend_n - start > 0):
            take = min(a.batch, pend_n - start)
            sl = slice(start, start + take)
            hf = arrs["hist_from"][sl][:, :n_ply] if n_ply > 0 else None
            ht = arrs["hist_to"][sl][:, :n_ply] if n_ply > 0 else None
            hc = arrs["hist_cap"][sl][:, :n_ply] if n_ply > 0 else None
            cls_np, sq_np = encode_batch(model, arrs["packed_pre"][sl], hf, ht, hc,
                                         n_ply, device, use_amp)
            buf = {"unit_id": arrs["unit_id"][sl], "split": arrs["split"][sl],
                   "src_row": arrs["src_row"][sl]}
            for c in PASS_COLS:
                buf[c] = arrs[c][sl]
            append(cls_np, sq_np, buf)
            start += take
            if take < a.batch:
                break
        # keep the leftover (< batch) rows
        if start < pend_n:
            for k in pend:
                pend[k] = [arrs[k][start:]]
            pend_n = pend_n - start
        else:
            for k in pend:
                pend[k] = []
            pend_n = 0

    sel_arr = np.asarray(sorted(sel_set), dtype=np.int64)
    t0 = time.time()
    for cstart in range(0, n_rows, a.chunk_rows):
        cend = min(cstart + a.chunk_rows, n_rows)
        uid_c = src["unit_id"][cstart:cend].astype(np.int64)
        split_c = src["split"][cstart:cend].astype(np.int64)

        in_sel = np.isin(uid_c, sel_arr)
        cand = np.where(in_sel)[0]
        if cand.size == 0:
            continue

        # sequential capping over candidate rows (deterministic, first-N per unit/split)
        accept_local = []
        for li in cand:
            u = int(uid_c[li]); s = int(split_c[li])
            key = (u, s)
            cur = counts.get(key, 0)
            if cur < caps.get(s, 0):
                counts[key] = cur + 1
                accept_local.append(li)
        if not accept_local:
            continue
        accept_local = np.asarray(accept_local, dtype=np.int64)

        # vectorized extraction of accepted rows
        gidx = cstart + accept_local
        pend["packed_pre"].append(src["packed_pre"][cstart:cend][accept_local])
        pend["hist_from"].append(src["hist_from"][cstart:cend][accept_local])
        pend["hist_to"].append(src["hist_to"][cstart:cend][accept_local])
        pend["hist_cap"].append(src["hist_cap"][cstart:cend][accept_local])
        pend["unit_id"].append(uid_c[accept_local].astype(np.int32))
        pend["split"].append(split_c[accept_local].astype(np.int8))
        pend["src_row"].append(gidx.astype(np.int64))
        for c in PASS_COLS:
            pend[c].append(src[c][cstart:cend][accept_local])
        # record source indices for the first n-check accepted positions
        base_cache_idx = n_written + pend_n
        for j, g in enumerate(gidx):
            ci = base_cache_idx + j
            if ci < a.n_check:
                check_pairs.append((int(ci), int(g)))
        pend_n += accept_local.size

        flush(force=False)
        done = cend
        print(f"  scanned {done:,}/{n_rows:,} rows  accepted={n_written + pend_n:,}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    flush(force=True)
    print(f"encode done: {n_written:,} rows written in {time.time() - t0:.0f}s")

    # selected-subset unit table
    sel_row_mask = np.isin(units["unit_id"].astype(np.int64), sel_arr)
    unit_cols = {k: units[k][sel_row_mask] for k in units.files}
    np.savez(a.units_out, **unit_cols)
    print(f"wrote unit subset table {a.units_out} ({int(sel_row_mask.sum())} units)")

    out.close()

    # ---- alignment self-check -------------------------------------------------
    print("\n=== self-check ===")
    with h5py.File(a.out, "r") as f:
        M = f["cls"].shape[0]
        print(f"M={M:,}  cls {f['cls'].shape} {f['cls'].dtype}  "
              f"squares {f['squares'].shape} {f['squares'].dtype}")
        assert f["squares"].shape == (M, 64, d), f["squares"].shape
        assert f["cls"].shape == (M, d), f["cls"].shape

        cuid = f["unit_id"][:]
        csplit = f["split"][:]
        n_distinct = len(np.unique(cuid))
        n_train = int((csplit == 0).sum())
        n_test = int((csplit == 1).sum())
        novel_mask = np.isin(cuid.astype(np.int64), np.asarray(sorted(novel_set), dtype=np.int64))
        n_novel_pos = int(novel_mask.sum())
        print(f"distinct units={n_distinct}  train={n_train:,}  test={n_test:,}  "
              f"novel-unit positions={n_novel_pos:,}")

        # Re-encode the recorded source rows (as one batch, matching the production
        # encode path) and compare with the cached features. We gate on cosine
        # similarity rather than a tight allclose: the features are stored float16 and
        # computed in bf16 autocast, so bit-exactness is not expected -- but a
        # misaligned row would give a near-random (low-cosine) match, so cosine >= 0.999
        # robustly proves each cached row lines up with its source position + label.
        def _cos(x, y):
            x = x.ravel().astype(np.float64); y = y.ravel().astype(np.float64)
            return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))

        cis = [ci for ci, _ in check_pairs]
        gs = [g for _, g in check_pairs]
        pk = np.stack([src["packed_pre"][g] for g in gs], axis=0)
        hf = np.stack([src["hist_from"][g][:n_ply] for g in gs], axis=0) if n_ply > 0 else None
        ht = np.stack([src["hist_to"][g][:n_ply] for g in gs], axis=0) if n_ply > 0 else None
        hc = np.stack([src["hist_cap"][g][:n_ply] for g in gs], axis=0) if n_ply > 0 else None
        cls_re, sq_re = encode_batch(model, pk, hf, ht, hc, n_ply, device, use_amp)

        ok = True
        for j, (ci, g) in enumerate(check_pairs):
            cls_cos = _cos(cls_re[j], f["cls"][ci])
            sq_cos = _cos(sq_re[j], f["squares"][ci])
            cls_max = float(np.abs(cls_re[j].astype(np.float32) - f["cls"][ci].astype(np.float32)).max())
            sq_max = float(np.abs(sq_re[j].astype(np.float32) - f["squares"][ci].astype(np.float32)).max())
            lbl_ok = (int(f["from_sq"][ci]) == int(src["from_sq"][g])
                      and int(f["to_sq"][ci]) == int(src["to_sq"][g])
                      and int(f["unit_id"][ci]) == int(src["unit_id"][g])
                      and int(f["src_row"][ci]) == int(g))
            feat_ok = cls_cos >= 0.999 and sq_cos >= 0.999
            if not (feat_ok and lbl_ok):
                ok = False
            print(f"  check cache[{ci}] <- src[{g}]  cls_cos={cls_cos:.6f} sq_cos={sq_cos:.6f} "
                  f"(maxabs cls={cls_max:.3f} sq={sq_max:.3f})  label_ok={lbl_ok}")
        assert ok, "ALIGNMENT SELF-CHECK FAILED"
        print("alignment self-check PASSED")

    src.close()
    print(f"\nDONE. M={M:,} positions / {n_distinct} units -> {a.out}")


if __name__ == "__main__":
    raise SystemExit(main())
