"""
Configure the HDF5 library before h5py/libhdf5 is loaded.

Parallel readers on Docker bind mounts / virtiofs / NFS often hit EIO when the default
POSIX byte-range locks are enabled. This must run before ``import h5py`` in the process
that will open files (including DataLoader worker processes — they re-import modules).
"""

from __future__ import annotations

import os


def apply_hdf5_read_safety_env() -> None:
    """If unset, disable HDF5 file locking for read-heavy multi-process workloads."""
    if os.environ.get("HDF5_USE_FILE_LOCKING", "").strip() == "":
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
