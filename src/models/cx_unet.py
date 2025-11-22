"""Complex-valued U-Net modules following Cole et al. (2020)."""

from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModReLU_Old(nn.Module):
    '''
    ModReLU Activation Function
    '''
    def __init__(self, bias: float = 0.0):
        super().__init__()
        self.ReLU = nn.ReLU()
        self.bias = nn.Parameter(torch.full((), bias))
    def forward(self, X):
        return self.ReLU(X.abs() + self.bias) * (X / (X.abs() + 1e-6))
    
class ModReLU(nn.Module):
    def __init__(self, init_bias: float = -0.2):
        super().__init__()
        self.bias = nn.Parameter(torch.full((), init_bias))

    def forward(self, x):
        mag = torch.abs(x) + 1e-8
        gate = torch.relu(mag + torch.clamp(self.bias.view(1, -1, 1, 1), max=0.0))
        return gate * (x / mag)

class Cardioid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        theta = torch.angle(x)
        gate = 0.5 * (1.0 + torch.cos(theta))
        return gate * x

class CReLU(nn.Module):
    '''
    CReLu Activation Function
    '''
    def __init__(self):
        super().__init__()
        self.ReLU = nn.ReLU()
    def forward(self, X):
        return self.ReLU(X.real) + 1j * self.ReLU(X.imag)
    

"""Pooling layers operating independently on real/imag components."""
class ComplexAvgPool2d(nn.Module):
    '''
    2d Complex Average Pooling Layer
    '''
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad, divisor_override=divisor_override)
    def forward(self, X):
        real_part = self.pool(X.real)
        imag_part = self.pool(X.imag)
        return real_part + 1j * imag_part
    
class ComplexMaxMagPool2d(nn.Module):
    '''
    2d Complex Max Pooling Useing Magnitude
    '''
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=True, ceil_mode=False)
        self.return_indices = return_indices # To preserve expected functionality
    def forward(self, X):
        pass #TODO: Implement this pooling method based on magnitude
        
class ComplexMaxCompPool2d(nn.Module):
    '''
    2d Complex Max Pooling Useing Individual Components
    '''
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding, dilation=dilation, ceil_mode=False)
    def forward(self, X):
        # Apply max pooling based on magnitude
        real_part = self.pool(X.real)
        imag_part = self.pool(X.imag)
        return real_part + 1j * imag_part
    

class ComplexBatchNorm2d(nn.Module):
    """
    2D Complex Batch Normalization Layer
    """
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        # Start with identity 2x2 affine for [Re, Im]
        gamma_start = torch.eye(2).unsqueeze(0).repeat(num_features, 1, 1) / 2**0.5
        beta_start = torch.zeros(num_features, 2)
        self.gamma = nn.Parameter(gamma_start)  # (C, 2, 2)
        self.beta  = nn.Parameter(beta_start)   # (C, 2)

        # Running stats as buffers
        running_cov = torch.eye(2).unsqueeze(0).repeat(num_features, 1, 1)  # (C, 2, 2)
        running_mean = torch.zeros(num_features, 2)                          # (C, 2)
        self.register_buffer("running_cov", running_cov)
        self.register_buffer("running_mean", running_mean)

    def cov_1d(self, x):
        # x: (2, N)
        return torch.cov(x)

    def forward(self, x):
        """
        https://arxiv.org/pdf/1705.09792
        Treat real and imaginary parts as 2D real vectors and normalize accordingly.
        """
        B, C, H, W = x.shape

        # (C, B, H, W) -> flatten to (C, N)
        x_c = x.permute(1, 0, 2, 3).contiguous()
        x_flat = x_c.view(C, -1)  # (C, N)

        # Stack real/imag as a 2D real vector: (C, 2, N)
        x_ri = torch.stack((x_flat.real, x_flat.imag), dim=1)

        V = self.running_cov        # (C, 2, 2)
        Ex = self.running_mean      # (C, 2)
        momentum = self.momentum

        if self.training:
            # Batch stats
            V_batch = torch.vmap(self.cov_1d)(x_ri)  # (C, 2, 2)
            Ex_batch = x_ri.mean(dim=-1)             # (C, 2)

            # Update running stats in-place
            with torch.no_grad():
                self.running_mean.mul_(1 - momentum).add_(momentum * Ex_batch)
                self.running_cov.mul_(1 - momentum).add_(momentum * V_batch)

            V = V_batch
            Ex = Ex_batch

        # Whitening: V^{-1/2} via eigendecomposition
        eigvals, eigvecs = torch.linalg.eigh(V)  # (C, 2), (C, 2, 2)
        V_inv_sqrt = eigvecs @ torch.diag_embed((eigvals + self.eps).rsqrt()) @ eigvecs.transpose(-1, -2)

        # Apply batch norm: x_tilde = V^{-1/2} (x - E[x])
        x_centered = x_ri - Ex.unsqueeze(-1)     # (C, 2, N)
        x_tilde = V_inv_sqrt @ x_centered        # (C, 2, N)

        # Affine transform: y = gamma x_tilde + beta
        y_ri = self.gamma @ x_tilde + self.beta.view(C, 2, 1)  # (C, 2, N)

        # Back to complex (B, C, H, W)
        y_ri = y_ri.view(C, 2, B, H, W)
        y = y_ri[:, 0] + 1j * y_ri[:, 1]         # (C, B, H, W)
        y = y.permute(1, 0, 2, 3).contiguous()   # (B, C, H, W)

        return y
    
import math

def complex_init_trabelsi_conv_like(conv_r: nn.Module,
                                    conv_i: nn.Module,
                                    criterion: str = "he"):
    """
    Trabelsi et al. (Deep Complex Networks, Sec. 3.6) style initializer
    for a pair of conv-like modules (Conv2d or ConvTranspose2d).

    conv_r, conv_i: real-valued modules representing Re and Im of one complex conv.
    criterion: "he" or "glorot"
    """
    w_r = conv_r.weight.data
    w_i = conv_i.weight.data

    # Handle 1D/2D/3D kernels; kernel_size can be int or tuple
    if isinstance(conv_r.kernel_size, int):
        k = (conv_r.kernel_size,)
    else:
        k = conv_r.kernel_size

    receptive_field = 1
    for s in k:
        receptive_field *= s

    # Take groups into account: each output channel only sees in_channels / groups
    groups = getattr(conv_r, "groups", 1)

    fan_in  = (conv_r.in_channels  / groups) * receptive_field
    fan_out = (conv_r.out_channels / groups) * receptive_field

    if criterion.lower() == "he":
        var = 2.0 / fan_in
    elif criterion.lower() in ["glorot", "xavier"]:
        var = 2.0 / (fan_in + fan_out)
    else:
        raise ValueError("criterion must be 'he' or 'glorot'")

    # For complex weights: Var(W) = E|W|^2 = 2 * sigma^2  ⇒  sigma^2 = var / 2
    sigma = math.sqrt(var / 2.0)

    # Rayleigh magnitude: r ~ Rayleigh(sigma)
    u = torch.rand_like(w_r)
    r = sigma * torch.sqrt(-2.0 * torch.log(1.0 - u + 1e-7))

    # Phase uniform in [-π, π]
    theta = (2.0 * math.pi) * torch.rand_like(w_r) - math.pi

    w_r[:] = r * torch.cos(theta)
    w_i[:] = r * torch.sin(theta)

    # Zero biases in the convs; you already have a separate complex bias param
    if conv_r.bias is not None:
        conv_r.bias.data.zero_()
    if conv_i.bias is not None:
        conv_i.bias.data.zero_()

    
class ComplexConv2d(nn.Module):
    '''
    2D complex convolution implemented via paired real-valued convolutions.
    Includes complex batch normalization
    '''
    def __init__(self, in_channels, out_channels, kernel_size, use_bn=True, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        super().__init__()
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode, device=device, dtype=dtype)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode, device=device, dtype=dtype)
        complex_init_trabelsi_conv_like(self.conv_real, self.conv_imag, criterion="he")
        # self.conv_real.weight.data.mul_(1/2**0.5)
        # self.conv_imag.weight.data.mul_(1/2**0.5)
        self.bn = ComplexBatchNorm2d(out_channels) if use_bn else None
        self.bias = nn.Parameter(torch.zeros((out_channels, 2))) if bias else None
    def forward(self, X):
        real_part = self.conv_real(X.real) - self.conv_imag(X.imag)
        imag_part = self.conv_real(X.imag) + self.conv_imag(X.real)
        if self.bias is not None:
            real_part = real_part + self.bias[:, 0].view(1, -1, 1, 1)
            imag_part = imag_part + self.bias[:, 1].view(1, -1, 1, 1)
        x = real_part + 1j * imag_part
        if self.bn:
            x = self.bn(x)
        return x
    
class ComplexConvTranspose2d(nn.Module):
    '''
    2d Complex Transposed Convolution Layer
    Includes complex batch normalization
    '''
    def __init__(self, in_channels, out_channels, kernel_size, use_bn=True, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        super().__init__()
        self.tconv_real = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode, device=device, dtype=dtype)
        self.tconv_imag = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode, device=device, dtype=dtype)
        complex_init_trabelsi_conv_like(self.tconv_real, self.tconv_imag, criterion="he")
        # self.tconv_real.weight.data.mul_(1/2**0.5)
        # self.tconv_imag.weight.data.mul_(1/2**0.5)
        self.bn = ComplexBatchNorm2d(out_channels) if use_bn else None
        self.bias = nn.Parameter(torch.zeros((out_channels, 2))) if bias else None
    def forward(self, X):
        real_part = self.tconv_real(X.real) - self.tconv_imag(X.imag)
        imag_part = self.tconv_real(X.imag) + self.tconv_imag(X.real)
        if self.bias is not None:
            real_part = real_part + self.bias[:, 0].view(1, -1, 1, 1)
            imag_part = imag_part + self.bias[:, 1].view(1, -1, 1, 1)
        x = real_part + 1j * imag_part
        if self.bn:
            x = self.bn(x)
        return x

class ComplexConvBlock(nn.Module):
    """Stack of complex convolutions with optional activation and batch norm between each layer."""

    def __init__(self, in_channels: int, out_channels: int, depth: int = 2, activation: nn.Module | None = None):
        super().__init__()
        self.conv1 = ComplexConv2d(in_channels, out_channels, kernel_size=3, padding="same")
        self.convs = nn.ModuleList([
            ComplexConv2d(out_channels, out_channels, kernel_size=3, padding="same")
            for _ in range(depth - 1)
        ])
        self.act = activation

    def forward(self, x):
        x = self.conv1(x)
        if self.act:
            x = self.act(x)
        for conv in self.convs:
            x = conv(x)
            if self.act:
                x = self.act(x)
        return x


class ComplexUnet(nn.Module):
    """
    Complex-valued U-Net as described by Cole et al.

    Mirrors the architecture of the real-valued baseline while preserving
    complex interactions via dedicated convolution, pooling, and activations.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: Sequence[int] = (64, 128, 256, 512, 1024),
        convs_per_block: int = 2,
        pooling_func: nn.Module | None = None,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.pool = pooling_func or ComplexMaxCompPool2d(2, stride=2)
        self.activation = activation or CReLU()
        self.features = list(features)

        enc_features = [in_channels] + self.features[:-1]
        self.enc_blocks = nn.ModuleList([
            ComplexConvBlock(enc_features[i], enc_features[i + 1], depth=convs_per_block, activation=self.activation)
            for i in range(len(enc_features)-1)
        ])
        self.skip_convs = nn.ModuleList([
            ComplexConv2d(self.features[i], self.features[i + 1], kernel_size=1)
            for i in range(len(self.features)-1)
        ])
        self.bottleneck = nn.Sequential(
            ComplexConv2d(self.features[-2], self.features[-1], kernel_size=3, padding=1),
            self.activation,
        )

        dec_features = self.features[::-1]
        self.up_blocks = nn.ModuleList([
            ComplexConvTranspose2d(dec_features[i], dec_features[i], kernel_size=2, stride=2)
            for i in range(len(dec_features)-1)
        ])
        self.dec_blocks = nn.ModuleList([
            ComplexConvBlock(dec_features[i], dec_features[i+1], depth=convs_per_block, activation=self.activation)
            for i in range(len(dec_features)-1)
        ])
        self.final_conv = nn.Sequential(
            ComplexConv2d(self.features[0], self.features[0], kernel_size=3, padding="same"),
            self.activation,
            ComplexConv2d(self.features[0], out_channels, use_bn=False, kernel_size=1),
        )


    def forward(self, x):
        # Encoder
        skips = []
        for enc_block, skip_conv in zip(self.enc_blocks, self.skip_convs):
            x = enc_block(x)
            skips.append(skip_conv(x))
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x) 
        skips = skips[::-1]  # reverse for decoder

        # Decoder
        for up_block, dec_block, skip in zip(self.up_blocks, self.dec_blocks, skips):
            x = up_block(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x_re = F.interpolate(x.real, size=skip.shape[-2:], mode='bilinear', align_corners=False)
                x_im = F.interpolate(x.imag, size=skip.shape[-2:], mode='bilinear', align_corners=False)
                x = x_re + 1j * x_im
            x = dec_block(x + skip)

        # Final Convolution
        x = self.final_conv(x)
        return x
