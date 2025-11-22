"""Mask generation utilities."""

import torch


def mask_1d_equispaced(N: int, accel: int = 4, acs: int = 24, offset: int = 0) -> torch.Tensor:
    """
    Build a 1D undersampling mask with contiguous ACS and equispaced outer lines.

    Args:
        N: Length of the phase-encode dimension.
        accel: Target acceleration factor (roughly 1 / sampling rate).
        acs: Width of the fully sampled auto-calibration region.
        offset: Phase offset for the equispaced outer lines.
    """
    mask = torch.zeros(N, dtype=torch.bool)
    c0 = (N - acs) // 2
    mask[c0 : c0 + acs] = True
    for i in range(N):
        if i < c0 or i >= c0 + acs:
            if ((i - offset) % accel) == 0:
                mask[i] = True
    return mask


class EquispacedMasker:
    """Callable wrapper for equispaced sampling masks with ACS."""

    def __init__(self, accel: int = 4, acs: int = 24, offset: int = 0):
        self.accel = accel
        self.acs = acs
        self.offset = offset

    def __call__(self, N: int) -> torch.Tensor:
        """Return a mask tensor sized for the provided k-space width."""
        return mask_1d_equispaced(N, self.accel, self.acs, self.offset)
