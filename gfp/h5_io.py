"""HDF5 layout for Global From Predictor (slim packed rows)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from jepa3.packed_board_codec import PACKED_BOARD_LEN, PACKED_LAYOUT_VERSION

GFP_FORMAT_ATTR = "gfp_format"
GFP_LAYOUT_VERSION_ATTR = "gfp_layout_version"
ROW_COUNT_ATTR = "row_count"

DATASET_PACKED_PRE = "packed_pre"
DATASET_FROM_LEGAL_U64 = "from_legal_u64"
DATASET_FROM_SQ = "from_sq"

DEFAULT_CHUNK = 4096
GFP_FORMAT_VERSION = 1


def gfp_h5_row_count(path: Path | str) -> int:
    with h5py.File(path, "r") as f:
        if int(f.attrs.get(GFP_FORMAT_ATTR, 0)) != GFP_FORMAT_VERSION:
            raise ValueError(f"not a gfp HDF5 (missing or wrong {GFP_FORMAT_ATTR!r}): {path}")
        return int(f[DATASET_PACKED_PRE].shape[0])


def assert_gfp_h5(path: Path | str) -> None:
    with h5py.File(path, "r") as f:
        if int(f.attrs.get(GFP_FORMAT_ATTR, 0)) != GFP_FORMAT_VERSION:
            raise ValueError(f"missing or wrong {GFP_FORMAT_ATTR}: {path}")
        if int(f.attrs.get(GFP_LAYOUT_VERSION_ATTR, 0)) != PACKED_LAYOUT_VERSION:
            raise ValueError(
                f"{GFP_LAYOUT_VERSION_ATTR} must match packed board layout {PACKED_LAYOUT_VERSION}: {path}"
            )
        for name in (DATASET_PACKED_PRE, DATASET_FROM_LEGAL_U64, DATASET_FROM_SQ):
            if name not in f:
                raise ValueError(f"gfp HDF5 missing dataset {name!r}: {path}")


class GfpH5Writer:
    """Append-only writer for gfp move rows (packed pre-move board + from legality + label)."""

    def __init__(self, path: Path, *, batch_size: int = 4096, chunk_rows: int | None = None) -> None:
        self.path = path.expanduser().resolve()
        self.batch_size = int(batch_size)
        self.chunk_rows = int(chunk_rows) if chunk_rows is not None else DEFAULT_CHUNK
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = h5py.File(self.path, "w")
        self._f.attrs[GFP_FORMAT_ATTR] = np.int32(GFP_FORMAT_VERSION)
        self._f.attrs[GFP_LAYOUT_VERSION_ATTR] = np.int32(PACKED_LAYOUT_VERSION)
        self._n = 0

        cr = (min(self.chunk_rows, self.batch_size * 4), PACKED_BOARD_LEN)

        self._d_pre = self._f.create_dataset(
            DATASET_PACKED_PRE,
            shape=(0, PACKED_BOARD_LEN),
            maxshape=(None, PACKED_BOARD_LEN),
            dtype=np.uint8,
            chunks=cr,
            compression=None,
        )
        ch1 = (min(self.chunk_rows, self.batch_size * 4),)
        self._d_from_u64 = self._f.create_dataset(
            DATASET_FROM_LEGAL_U64,
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint64,
            chunks=ch1,
            compression=None,
        )
        self._d_fs = self._f.create_dataset(
            DATASET_FROM_SQ,
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint8,
            chunks=ch1,
            compression=None,
        )

        self._buf_pre: list[np.ndarray] = []
        self._buf_u64: list[np.uint64] = []
        self._buf_fs: list[np.uint8] = []

    def append_row(
        self,
        *,
        packed_pre: np.ndarray,
        from_legal_u64: int | np.uint64,
        from_sq: int,
    ) -> None:
        self._buf_pre.append(np.asarray(packed_pre, dtype=np.uint8).reshape(PACKED_BOARD_LEN))
        self._buf_u64.append(np.uint64(from_legal_u64))
        self._buf_fs.append(np.uint8(from_sq))
        if len(self._buf_pre) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._buf_pre:
            return
        m = len(self._buf_pre)
        o = self._n

        block_pre = np.stack(self._buf_pre, axis=0)
        block_u64 = np.asarray(self._buf_u64, dtype=np.uint64)
        block_fs = np.asarray(self._buf_fs, dtype=np.uint8)

        for d, arr in (
            (self._d_pre, block_pre),
            (self._d_from_u64, block_u64),
            (self._d_fs, block_fs),
        ):
            if d.ndim == 2:
                d.resize((o + m, PACKED_BOARD_LEN))
            else:
                d.resize((o + m,))
            d[o : o + m] = arr

        self._buf_pre.clear()
        self._buf_u64.clear()
        self._buf_fs.clear()
        self._n += m

    def close(self) -> None:
        self.flush()
        self._f.attrs[ROW_COUNT_ATTR] = np.int64(self._n)
        self._f.close()

    def __enter__(self) -> GfpH5Writer:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
