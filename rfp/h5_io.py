"""HDF5 layout for Residual From Predictor (precomputed delta-z history + labels)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from jepa3.packed_board_codec import PACKED_LAYOUT_VERSION

RFP_FORMAT_ATTR = "rfp_format"
RFP_LAYOUT_VERSION_ATTR = "rfp_layout_version"
ROW_COUNT_ATTR = "row_count"
ATTR_HISTORY_LEN = "history_len"
ATTR_D_MODEL = "d_model"

DATASET_DELTA_Z = "delta_z"
DATASET_Z_CURR = "z_curr"
DATASET_HISTORY_MASK = "history_mask"
DATASET_FROM_LEGAL_U64 = "from_legal_u64"
DATASET_FROM_SQ = "from_sq"
DATASET_ELO_BUCKET = "elo_bucket"

DEFAULT_CHUNK = 4096
RFP_FORMAT_VERSION = 1
RFP_LAYOUT_VERSION = 1


def rfp_h5_row_count(path: Path | str) -> int:
    with h5py.File(path, "r") as f:
        if int(f.attrs.get(RFP_FORMAT_ATTR, 0)) != RFP_FORMAT_VERSION:
            raise ValueError(f"not an rfp HDF5 (missing or wrong {RFP_FORMAT_ATTR!r}): {path}")
        return int(f[DATASET_DELTA_Z].shape[0])


def rfp_h5_attrs(path: Path | str) -> tuple[int, int]:
    """Returns (history_len, d_model)."""
    with h5py.File(path, "r") as f:
        if int(f.attrs.get(RFP_FORMAT_ATTR, 0)) != RFP_FORMAT_VERSION:
            raise ValueError(f"not an rfp HDF5: {path}")
        n = int(f.attrs[ATTR_HISTORY_LEN])
        d = int(f.attrs[ATTR_D_MODEL])
        return n, d


def assert_rfp_h5(path: Path | str) -> None:
    with h5py.File(path, "r") as f:
        if int(f.attrs.get(RFP_FORMAT_ATTR, 0)) != RFP_FORMAT_VERSION:
            raise ValueError(f"missing or wrong {RFP_FORMAT_ATTR}: {path}")
        if int(f.attrs.get(RFP_LAYOUT_VERSION_ATTR, 0)) != RFP_LAYOUT_VERSION:
            raise ValueError(f"missing or wrong {RFP_LAYOUT_VERSION_ATTR}: {path}")
        if int(f.attrs.get("packed_layout_version", 0)) != PACKED_LAYOUT_VERSION:
            raise ValueError(f"packed_layout_version must be {PACKED_LAYOUT_VERSION}: {path}")
        hl = int(f.attrs[ATTR_HISTORY_LEN])
        dm = int(f.attrs[ATTR_D_MODEL])
        if DATASET_DELTA_Z not in f:
            raise ValueError(f"rfp HDF5 missing dataset {DATASET_DELTA_Z!r}: {path}")
        ds_dz = f[DATASET_DELTA_Z]
        if len(ds_dz.shape) != 3 or tuple(ds_dz.shape[1:]) != (hl, dm):
            raise ValueError(
                f"{DATASET_DELTA_Z} shape {ds_dz.shape} incompatible with attrs "
                f"history_len={hl}, d_model={dm}: {path}"
            )
        if DATASET_Z_CURR not in f:
            raise ValueError(f"rfp HDF5 missing dataset {DATASET_Z_CURR!r}: {path}")
        ds_z = f[DATASET_Z_CURR]
        if len(ds_z.shape) != 2 or int(ds_z.shape[1]) != dm:
            raise ValueError(f"{DATASET_Z_CURR} d_model mismatch: {path}")
        if DATASET_HISTORY_MASK not in f:
            raise ValueError(f"rfp HDF5 missing dataset {DATASET_HISTORY_MASK!r}: {path}")
        ds_hm = f[DATASET_HISTORY_MASK]
        if len(ds_hm.shape) != 2 or int(ds_hm.shape[1]) != hl:
            raise ValueError(f"{DATASET_HISTORY_MASK} history_len mismatch: {path}")
        for name in (DATASET_FROM_LEGAL_U64, DATASET_FROM_SQ, DATASET_ELO_BUCKET):
            if name not in f:
                raise ValueError(f"rfp HDF5 missing dataset {name!r}: {path}")


class RfpH5Writer:
    """Append-only writer for rfp rows (float16 embeddings + gfp-compatible labels)."""

    def __init__(
        self,
        path: Path,
        *,
        history_len: int,
        d_model: int,
        batch_size: int = 4096,
        chunk_rows: int | None = None,
    ) -> None:
        self.path = path.expanduser().resolve()
        self.history_len = int(history_len)
        self.d_model = int(d_model)
        self.batch_size = int(batch_size)
        self.chunk_rows = int(chunk_rows) if chunk_rows is not None else DEFAULT_CHUNK
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = h5py.File(self.path, "w")
        self._f.attrs[RFP_FORMAT_ATTR] = np.int32(RFP_FORMAT_VERSION)
        self._f.attrs[RFP_LAYOUT_VERSION_ATTR] = np.int32(RFP_LAYOUT_VERSION)
        self._f.attrs["packed_layout_version"] = np.int32(PACKED_LAYOUT_VERSION)
        self._f.attrs[ATTR_HISTORY_LEN] = np.int32(self.history_len)
        self._f.attrs[ATTR_D_MODEL] = np.int32(self.d_model)
        self._n = 0

        cr = (
            min(self.chunk_rows, self.batch_size * 4),
            self.history_len,
            self.d_model,
        )
        cr2 = (min(self.chunk_rows, self.batch_size * 4), self.d_model)
        cr1 = (min(self.chunk_rows, self.batch_size * 4), self.history_len)
        ch1 = (min(self.chunk_rows, self.batch_size * 4),)

        self._d_dz = self._f.create_dataset(
            DATASET_DELTA_Z,
            shape=(0, self.history_len, self.d_model),
            maxshape=(None, self.history_len, self.d_model),
            dtype=np.float16,
            chunks=cr,
            compression=None,
        )
        self._d_zc = self._f.create_dataset(
            DATASET_Z_CURR,
            shape=(0, self.d_model),
            maxshape=(None, self.d_model),
            dtype=np.float16,
            chunks=cr2,
            compression=None,
        )
        self._d_hm = self._f.create_dataset(
            DATASET_HISTORY_MASK,
            shape=(0, self.history_len),
            maxshape=(None, self.history_len),
            dtype=np.uint8,
            chunks=cr1,
            compression=None,
        )
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
        self._d_elo = self._f.create_dataset(
            DATASET_ELO_BUCKET,
            shape=(0,),
            maxshape=(None,),
            dtype=np.int16,
            chunks=ch1,
            compression=None,
        )

        self._buf_dz: list[np.ndarray] = []
        self._buf_zc: list[np.ndarray] = []
        self._buf_hm: list[np.ndarray] = []
        self._buf_u64: list[np.uint64] = []
        self._buf_fs: list[np.uint8] = []
        self._buf_elo: list[np.int16] = []

    def append_row(
        self,
        *,
        delta_z: np.ndarray,
        z_curr: np.ndarray,
        history_mask: np.ndarray,
        from_legal_u64: int | np.uint64,
        from_sq: int,
        elo_bucket: int,
    ) -> None:
        dz = np.asarray(delta_z, dtype=np.float16).reshape(self.history_len, self.d_model)
        zc = np.asarray(z_curr, dtype=np.float16).reshape(self.d_model)
        hm = np.asarray(history_mask, dtype=np.uint8).reshape(self.history_len)
        self._buf_dz.append(dz)
        self._buf_zc.append(zc)
        self._buf_hm.append(hm)
        self._buf_u64.append(np.uint64(from_legal_u64))
        self._buf_fs.append(np.uint8(from_sq))
        self._buf_elo.append(np.int16(elo_bucket))
        if len(self._buf_dz) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._buf_dz:
            return
        m = len(self._buf_dz)
        o = self._n

        block_dz = np.stack(self._buf_dz, axis=0)
        block_zc = np.stack(self._buf_zc, axis=0)
        block_hm = np.stack(self._buf_hm, axis=0)
        block_u64 = np.asarray(self._buf_u64, dtype=np.uint64)
        block_fs = np.asarray(self._buf_fs, dtype=np.uint8)
        block_elo = np.asarray(self._buf_elo, dtype=np.int16)

        self._d_dz.resize((o + m, self.history_len, self.d_model))
        self._d_zc.resize((o + m, self.d_model))
        self._d_hm.resize((o + m, self.history_len))
        self._d_from_u64.resize((o + m,))
        self._d_fs.resize((o + m,))
        self._d_elo.resize((o + m,))

        self._d_dz[o : o + m] = block_dz
        self._d_zc[o : o + m] = block_zc
        self._d_hm[o : o + m] = block_hm
        self._d_from_u64[o : o + m] = block_u64
        self._d_fs[o : o + m] = block_fs
        self._d_elo[o : o + m] = block_elo

        self._buf_dz.clear()
        self._buf_zc.clear()
        self._buf_hm.clear()
        self._buf_u64.clear()
        self._buf_fs.clear()
        self._buf_elo.clear()
        self._n += m

    def close(self) -> None:
        self.flush()
        self._f.attrs[ROW_COUNT_ATTR] = np.int64(self._n)
        self._f.close()

    def __enter__(self) -> RfpH5Writer:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
