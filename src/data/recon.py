"""Reconstruction helpers (IFFT)."""

import torch


def ifft2c(kspace: torch.Tensor) -> torch.Tensor:
    """2D centered inverse FFT for reconstructing images from k-space."""
    return torch.fft.ifftshift(
        torch.fft.ifft2(torch.fft.fftshift(kspace, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )
