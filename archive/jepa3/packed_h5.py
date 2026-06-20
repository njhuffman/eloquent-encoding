"""HDF5 layout for jepa3 packed move rows (fixed-width, no gzip on hot columns)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from jepa3.packed_board_codec import PACKED_BOARD_LEN, PACKED_LAYOUT_VERSION

JEPA3_PACKED_FORMAT_ATTR = "jepa3_packed_format"
ROW_COUNT_ATTR = "row_count"

DATASET_PACKED_PRE = "packed_pre"
DATASET_PACKED_POST = "packed_post"
DATASET_FROM_LEGAL_U64 = "from_legal_u64"
DATASET_TO_LEGAL_U64 = "to_legal_u64"
DATASET_FROM_SQ = "from_sq"
DATASET_TO_SQ = "to_sq"
DATASET_PROMOTION = "promotion"
DATASET_ELO = "elo_to_move"

DEFAULT_CHUNK = 4096


def packed_h5_row_count(path: Path | str) -> int:
    with h5py.File(path, "r") as f:
        if JEPA3_PACKED_FORMAT_ATTR not in f.attrs:
            raise ValueError(f"not a jepa3 packed HDF5 (missing {JEPA3_PACKED_FORMAT_ATTR!r}): {path}")
        d = f[DATASET_PACKED_PRE]
        return int(d.shape[0])


def assert_packed_h5(path: Path | str) -> None:
    with h5py.File(path, "r") as f:
        if int(f.attrs.get(JEPA3_PACKED_FORMAT_ATTR, 0)) != 1:
            raise ValueError(f"missing or wrong {JEPA3_PACKED_FORMAT_ATTR}: {path}")
        for name in (
            DATASET_PACKED_PRE,
            DATASET_PACKED_POST,
            DATASET_FROM_LEGAL_U64,
            DATASET_TO_LEGAL_U64,
            DATASET_FROM_SQ,
            DATASET_TO_SQ,
            DATASET_PROMOTION,
            DATASET_ELO,
        ):
            if name not in f:
                raise ValueError(f"packed HDF5 missing dataset {name!r}: {path}")


class PackedMoveH5Writer:
    """Append-only writer; mirrors dataset_generation SampleBatchWriter flush pattern."""

    def __init__(self, path: Path, *, batch_size: int = 4096, chunk_rows: int | None = None) -> None:
        self.path = path.expanduser().resolve()
        self.batch_size = int(batch_size)
        self.chunk_rows = int(chunk_rows) if chunk_rows is not None else DEFAULT_CHUNK
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = h5py.File(self.path, "w")
        self._f.attrs[JEPA3_PACKED_FORMAT_ATTR] = np.int32(1)
        self._f.attrs["packed_layout_version"] = np.int32(PACKED_LAYOUT_VERSION)
        self._n = 0

        cr = (min(self.chunk_rows, self.batch_size * 4), PACKED_BOARD_LEN)

        def _create_2d(name: str, dtype: np.dtype) -> h5py.Dataset:
            return self._f.create_dataset(
                name,
                shape=(0, PACKED_BOARD_LEN),
                maxshape=(None, PACKED_BOARD_LEN),
                dtype=dtype,
                chunks=cr,
                compression=None,
            )

        self._d_pre = _create_2d(DATASET_PACKED_PRE, np.uint8)
        self._d_post = _create_2d(DATASET_PACKED_POST, np.uint8)

        ch1 = (min(self.chunk_rows, self.batch_size * 4),)
        self._d_from_u64 = self._f.create_dataset(
            DATASET_FROM_LEGAL_U64,
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint64,
            chunks=ch1,
            compression=None,
        )
        self._d_to_u64 = self._f.create_dataset(
            DATASET_TO_LEGAL_U64,
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint64,
            chunks=ch1,
            compression=None,
        )
        u8_chunks = ch1
        self._d_fs = self._f.create_dataset(
            DATASET_FROM_SQ,
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint8,
            chunks=u8_chunks,
            compression=None,
        )
        self._d_ts = self._f.create_dataset(
            DATASET_TO_SQ,
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint8,
            chunks=u8_chunks,
            compression=None,
        )
        self._d_pr = self._f.create_dataset(
            DATASET_PROMOTION,
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint8,
            chunks=u8_chunks,
            compression=None,
        )
        self._d_elo = self._f.create_dataset(
            DATASET_ELO,
            shape=(0,),
            maxshape=(None,),
            dtype=np.int16,
            chunks=u8_chunks,
            compression=None,
        )

        self._buf: dict[str, list] = {
            DATASET_PACKED_PRE: [],
            DATASET_PACKED_POST: [],
            DATASET_FROM_LEGAL_U64: [],
            DATASET_TO_LEGAL_U64: [],
            DATASET_FROM_SQ: [],
            DATASET_TO_SQ: [],
            DATASET_PROMOTION: [],
            DATASET_ELO: [],
        }

    def append_row(
        self,
        *,
        packed_pre: np.ndarray,
        packed_post: np.ndarray,
        from_legal_u64: int | np.uint64,
        to_legal_u64: int | np.uint64,
        from_sq: int,
        to_sq: int,
        promotion: int,
        elo_to_move: int,
    ) -> None:
        self._buf[DATASET_PACKED_PRE].append(np.asarray(packed_pre, dtype=np.uint8).reshape(PACKED_BOARD_LEN))
        self._buf[DATASET_PACKED_POST].append(np.asarray(packed_post, dtype=np.uint8).reshape(PACKED_BOARD_LEN))
        self._buf[DATASET_FROM_LEGAL_U64].append(np.uint64(from_legal_u64))
        self._buf[DATASET_TO_LEGAL_U64].append(np.uint64(to_legal_u64))
        self._buf[DATASET_FROM_SQ].append(np.uint8(from_sq))
        self._buf[DATASET_TO_SQ].append(np.uint8(to_sq))
        self._buf[DATASET_PROMOTION].append(np.uint8(promotion))
        self._buf[DATASET_ELO].append(np.int16(elo_to_move))
        if len(self._buf[DATASET_PACKED_PRE]) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._buf[DATASET_PACKED_PRE]:
            return
        m = len(self._buf[DATASET_PACKED_PRE])
        o = self._n

        block_pre = np.stack(self._buf[DATASET_PACKED_PRE], axis=0)
        block_post = np.stack(self._buf[DATASET_PACKED_POST], axis=0)
        for d, arr in (
            (self._d_pre, block_pre),
            (self._d_post, block_post),
            (self._d_from_u64, np.asarray(self._buf[DATASET_FROM_LEGAL_U64], dtype=np.uint64)),
            (self._d_to_u64, np.asarray(self._buf[DATASET_TO_LEGAL_U64], dtype=np.uint64)),
            (self._d_fs, np.asarray(self._buf[DATASET_FROM_SQ], dtype=np.uint8)),
            (self._d_ts, np.asarray(self._buf[DATASET_TO_SQ], dtype=np.uint8)),
            (self._d_pr, np.asarray(self._buf[DATASET_PROMOTION], dtype=np.uint8)),
            (self._d_elo, np.asarray(self._buf[DATASET_ELO], dtype=np.int16)),
        ):
            if d.ndim == 2:
                d.resize((o + m, PACKED_BOARD_LEN))
            else:
                d.resize((o + m,))
            d[o : o + m] = arr

        for k in self._buf:
            self._buf[k].clear()
        self._n += m

    def close(self) -> None:
        self.flush()
        self._f.attrs[ROW_COUNT_ATTR] = np.int64(self._n)
        self._f.close()

    def __enter__(self) -> PackedMoveH5Writer:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
