"""Real-valued U-Net baseline mirroring the complex architecture from Cole et al."""

from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def complex_to_channels(x: torch.Tensor) -> torch.Tensor:
    """Convert a complex tensor (B, C, H, W) to a 2C-channel real tensor."""
    if not torch.is_complex(x):
        raise ValueError("Expected complex tensor with dtype=torch.complex*")
    return torch.cat([x.real, x.imag], dim=1)


def channels_to_complex(x: torch.Tensor) -> torch.Tensor:
    """Convert a real tensor with 2C channels back to a complex tensor."""
    if x.shape[1] % 2 != 0:
        raise ValueError("Channel dimension must be even to form complex pairs.")
    real, imag = torch.chunk(x, 2, dim=1)
    return torch.complex(real, imag)

class ConvLayer2d(nn.Module):
    '''
    2D  convolution followed by optional BatchNorm
    '''
    def __init__(self, in_channels, out_channels, kernel_size, use_bn=True, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None
    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        return x

class ConvTransposeLayer2d(nn.Module):
    '''
    2D  convolution transpose followed by optional BatchNorm
    '''
    def __init__(self, in_channels, out_channels, kernel_size, use_bn=True, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None
    def forward(self, x):
        x = self.tconv(x)
        if self.bn:
            x = self.bn(x)
        return x


class RealConvBlock(nn.Module):
    """Two-channel analogue of ComplexConvBlock with optional activation."""

    def __init__(self, in_channels: int, out_channels: int, depth: int = 2, activation: nn.Module | None = None):
        super().__init__()
        self.conv1 = ConvLayer2d(in_channels, out_channels, kernel_size=3, padding="same")
        self.convs = nn.ModuleList([
            ConvLayer2d(out_channels, out_channels, kernel_size=3, padding="same")
            for _ in range(depth - 1)
        ])
        self.act = activation or nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        if self.act:
            x = self.act(x)
        for conv in self.convs:
            x = conv(x)
            if self.act:
                x = self.act(x)
        return x


class RealUnet(nn.Module):
    """Real-valued U-Net baseline for the Cole et al. complex-vs-real study."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: Sequence[int] = (64, 128, 256, 512, 1024),
        convs_per_block: int = 2,
        activation: nn.Module | None = None,
        width_scale: float = 0.5,
    ) -> None:
        """
        Args:
            in_channels: Number of complex-valued channels (typically 1).
            out_channels: Number of complex-valued output channels (typically 1).
            features: Feature widths used by the complex network. Real widths are scaled by `width_scale`.
            convs_per_block: Number of conv layers per U-Net block.
            activation: Activation module; defaults to ReLU.
            width_scale: Multiplier applied to `features` to match the parameter count
                of the complex-valued network (0.5 ≈ same params because complex conv has ~2x weights).
        """
        super().__init__()
        self.activation = activation or nn.ReLU(inplace=True)
        self.in_channels = in_channels * 2  # real+imag stacked
        self.out_channels = out_channels * 2
        self.features = [max(1, int(f * width_scale)) for f in features]

        enc_features: List[int] = [self.in_channels] + self.features[:-1]
        self.enc_blocks = nn.ModuleList([
            RealConvBlock(enc_features[i], enc_features[i + 1], depth=convs_per_block, activation=self.activation)
            for i in range(len(enc_features) - 1)
        ])
        self.skip_convs = nn.ModuleList([
            ConvLayer2d(self.features[i], self.features[i + 1], kernel_size=1)
            for i in range(len(self.features) - 1)
        ])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(
            ConvLayer2d(self.features[-2], self.features[-1], kernel_size=3, padding="same"),
            self.activation,
        )

        dec_features = self.features[::-1]
        self.up_blocks = nn.ModuleList([
            ConvTransposeLayer2d(dec_features[i], dec_features[i], kernel_size=2, stride=2)
            for i in range(len(dec_features) - 1)
        ])
        self.dec_blocks = nn.ModuleList([
            RealConvBlock(dec_features[i], dec_features[i + 1], depth=convs_per_block, activation=self.activation)
            for i in range(len(dec_features) - 1)
        ])
        self.final_conv = nn.Sequential(
            ConvLayer2d(self.features[0], self.features[0], kernel_size=3, padding="same"),
            self.activation,
            ConvLayer2d(self.features[0], self.out_channels, use_bn=False, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that returns a complex-valued image tensor."""
        x_real = complex_to_channels(x)
        skips: list[torch.Tensor] = []

        for enc_block, skip_conv in zip(self.enc_blocks, self.skip_convs):
            x_real = enc_block(x_real)
            skips.append(skip_conv(x_real))
            x_real = self.pool(x_real)

        x_real = self.bottleneck(x_real)
        skips = skips[::-1]

        for up_block, dec_block, skip in zip(self.up_blocks, self.dec_blocks, skips):
            x_real = up_block(x_real)
            if x_real.shape[-2:] != skip.shape[-2:]:
                x_real = F.interpolate(x_real, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x_real = dec_block(x_real + skip)

        x_real = self.final_conv(x_real)
        return channels_to_complex(x_real)


def match_real_feature_list(complex_features: Iterable[int], width_scale: float = 0.5) -> List[int]:
    """Utility to compute real-valued feature widths for parameter parity."""
    return [max(1, int(f * width_scale)) for f in complex_features]
