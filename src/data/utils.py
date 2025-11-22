"""Utility helpers for data IO."""

import h5py
import torch


def load_h5_file(filepath: str) -> dict:
    """Load an HDF5 file from fastMRI and return a dictionary with tensors."""
    with h5py.File(filepath, "r") as f_read:
        ds = {"kspace": torch.tensor(f_read["kspace"][()])}
    return ds
