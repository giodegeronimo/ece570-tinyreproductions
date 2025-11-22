"""fastMRI dataset utilities."""

import os
from typing import Callable, List, Sequence, Tuple

import torch
import torchvision
from torch.utils.data import Dataset

from .masking import EquispacedMasker
from .recon import ifft2c
from .utils import load_h5_file


class SingleCoilDataset(Dataset):
    """Slice-level dataset that loads fastMRI single-coil HDF5 files."""

    def __init__(self, folder_path: str, mask_func: Callable = EquispacedMasker()):
        """
        Build an index of (filepath, slice_idx) pairs for per-slice loading.

        Args:
            folder_path: Directory containing fastMRI .h5 volumes.
            mask_func: Callable producing a 1D sampling mask for k-space width.
        """

        self.filepaths: List[str] = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".h5")
        ]
        self.mask_func = mask_func

        # Pre-compute slice indices so __getitem__ loads only the needed slice.
        filepath_slice_idxs: List[Tuple[str, int]] = []
        for filepath in self.filepaths:
            ds = load_h5_file(filepath)
            n_slices = ds["kspace"].shape[-3]
            for slice_idx in range(n_slices):
                filepath_slice_idxs.append((filepath, slice_idx))
        self.filepath_slice_idxs: Sequence[Tuple[str, int]] = filepath_slice_idxs

        # Normalize image sizes via center crop (fastMRI single-coil knee varies by scanner).
        self.crop = torchvision.transforms.CenterCrop((320, 256)) #((640, 320))

    def __len__(self) -> int:
        """Return number of slice samples across the dataset."""
        return len(self.filepath_slice_idxs)

    def __getitem__(self, idx: int):
        """Load a masked and full reconstruction pair for a given slice index."""
        filepath, slice_idx = self.filepath_slice_idxs[idx]
        ds = load_h5_file(filepath)
        kspace = ds["kspace"][slice_idx, :, :]
        mask = self.mask_func(kspace.shape[-1])  # mask along the last dimension
        masked_kspace = kspace * mask
        image_full = ifft2c(kspace)
        image_masked = ifft2c(masked_kspace)
        image_masked = image_masked.unsqueeze(0)  # add channel dimension
        image_full = image_full.unsqueeze(0)

        # Normalize to max abs value in full image.
        scale = torch.max(image_full.abs()).clamp_min(1e-8)
        image_full = image_full / scale
        image_masked = image_masked / scale

        # Apply center crop for consistent geometry.
        image_masked = self.crop(image_masked)
        image_full = self.crop(image_full)

        return image_masked, image_full
