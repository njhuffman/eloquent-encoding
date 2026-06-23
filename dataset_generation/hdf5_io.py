from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

CHUNK = 8192
PACKED_LEN = 34


class PackedBatchWriter:
    """Append rows to a new HDF5 in the packed training schema (+ opp_elo, result)."""

    _SCALAR = (
        ("from_legal_u64", np.uint64), ("to_legal_u64", np.uint64),
        ("from_sq", np.uint8), ("to_sq", np.uint8), ("promotion", np.uint8),
        ("elo_to_move", np.int16), ("opp_elo", np.int16), ("result", np.int8),
    )
    COLUMNS = ("packed_pre",) + tuple(n for n, _ in _SCALAR)

    def __init__(self, path: Path, batch_size: int = 1024) -> None:
        self.path = path
        self.batch_size = batch_size
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = h5py.File(path, "w")
        self._n = 0
        # Uncompressed: training shuffles rows, so random access must not pay per-chunk
        # gzip decompression (gzip made the dataloader CPU-bound and starved the GPU).
        # Matches the original j3 packed datasets, which are uncompressed.
        self._f.create_dataset("packed_pre", shape=(0, PACKED_LEN), maxshape=(None, PACKED_LEN),
                               dtype=np.uint8, chunks=(CHUNK, PACKED_LEN))
        for name, dt in self._SCALAR:
            self._f.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dt, chunks=(CHUNK,))
        self._buf = {c: [] for c in self.COLUMNS}

    def __enter__(self) -> "PackedBatchWriter":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self.flush()
        self._f.attrs["row_count"] = self._n
        self._f.close()

    def flush(self) -> None:
        if not self._buf["packed_pre"]:
            return
        m = len(self._buf["packed_pre"])
        o = self._n
        for name in self.COLUMNS:
            d = self._f[name]
            if name == "packed_pre":
                d.resize((o + m, PACKED_LEN))
                d[o : o + m] = np.asarray(self._buf[name], dtype=np.uint8)
            else:
                d.resize((o + m,))
                d[o : o + m] = np.asarray(self._buf[name], dtype=d.dtype)
            self._buf[name].clear()
        self._n += m

    def append_row(self, *, packed_pre, from_legal_u64, to_legal_u64, from_sq, to_sq,
                   promotion, elo_to_move, opp_elo, result) -> None:
        self._buf["packed_pre"].append(np.asarray(packed_pre, dtype=np.uint8).reshape(PACKED_LEN))
        self._buf["from_legal_u64"].append(np.uint64(from_legal_u64))
        self._buf["to_legal_u64"].append(np.uint64(to_legal_u64))
        self._buf["from_sq"].append(from_sq)
        self._buf["to_sq"].append(to_sq)
        self._buf["promotion"].append(promotion)
        self._buf["elo_to_move"].append(elo_to_move)
        self._buf["opp_elo"].append(opp_elo)
        self._buf["result"].append(result)
        if len(self._buf["packed_pre"]) >= self.batch_size:
            self.flush()


class SampleBatchWriter:
    """Append rows to a new HDF5 file with fixed column layout."""

    COLUMNS = (
        "fen",
        "side_to_move",
        "elo_to_move",
        "from_sq",
        "to_sq",
        "promotion",
        "source_plan_index",
        "stratum_index",
    )

    def __init__(self, path: Path, batch_size: int = 1024) -> None:
        self.path = path
        self.batch_size = batch_size
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = h5py.File(path, "w")
        self._n = 0
        str_dt = h5py.string_dtype(encoding="utf-8")
        self._f.create_dataset(
            "fen",
            shape=(0,),
            maxshape=(None,),
            dtype=str_dt,
            chunks=(CHUNK,),
            compression="gzip",
            compression_opts=4,
        )
        for name, dt in (
            ("side_to_move", np.uint8),
            ("elo_to_move", np.int16),
            ("from_sq", np.uint8),
            ("to_sq", np.uint8),
            ("promotion", np.uint8),
            ("source_plan_index", np.uint16),
            ("stratum_index", np.uint16),
        ):
            self._f.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                dtype=dt,
                chunks=(CHUNK,),
                compression="gzip",
                compression_opts=4,
            )
        self._buf = {c: [] for c in self.COLUMNS}

    def close(self) -> None:
        self.flush()
        self._f.attrs["row_count"] = self._n
        self._f.close()

    def __enter__(self) -> SampleBatchWriter:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def flush(self) -> None:
        if not self._buf["fen"]:
            return
        m = len(self._buf["fen"])
        for name in self.COLUMNS:
            block = self._buf[name]
            d = self._f[name]
            o = self._n
            d.resize((o + m,))
            d[o : o + m] = np.asarray(block, dtype=d.dtype)
            self._buf[name].clear()
        self._n += m

    def append_row(
        self,
        *,
        fen: str,
        side_to_move: int,
        elo_to_move: int,
        from_sq: int,
        to_sq: int,
        promotion: int,
        source_plan_index: int,
        stratum_index: int,
    ) -> None:
        self._buf["fen"].append(fen)
        self._buf["side_to_move"].append(side_to_move)
        self._buf["elo_to_move"].append(elo_to_move)
        self._buf["from_sq"].append(from_sq)
        self._buf["to_sq"].append(to_sq)
        self._buf["promotion"].append(promotion)
        self._buf["source_plan_index"].append(source_plan_index)
        self._buf["stratum_index"].append(stratum_index)
        if len(self._buf["fen"]) >= self.batch_size:
            self.flush()
