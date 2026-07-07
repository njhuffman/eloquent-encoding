"""Concatenate several h5 shard files (identical schema) into one, field by field."""
import sys, h5py, numpy as np

out, ins = sys.argv[1], sys.argv[2:]
keys = list(h5py.File(ins[0], "r").keys())
fo = h5py.File(out, "w")
for k in keys:
    parts = [h5py.File(i, "r") for i in ins]
    total = sum(p[k].shape[0] for p in parts)
    sh = parts[0][k].shape[1:]
    d = fo.create_dataset(k, shape=(total, *sh), dtype=parts[0][k].dtype, chunks=(min(8192, total), *sh))
    off = 0
    for p in parts:
        n = p[k].shape[0]
        if n: d[off:off + n] = p[k][:]
        off += n
    print(f"{k}: {total:,}", flush=True)
fo.close()
print(f"merged {len(ins)} shards -> {out}")
