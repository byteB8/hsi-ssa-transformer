"""HybridSN: a hybrid 3D/2D convolutional network for hyperspectral classification.

Roy et al., "HybridSN: Exploring 3-D-2-D CNN Feature Hierarchy for Hyperspectral
Image Classification", IEEE GRSL 2020 (arXiv:1902.06701).

Three 3D convolutions extract joint spectral-spatial structure, the spectral and
channel axes are then folded together and a 2D convolution extracts the
remaining spatial structure. ``use_cbam`` inserts a Convolutional Block
Attention Module (Woo et al., ECCV 2018) after each 3D stage, giving the network
a learned reweighting over feature channels and spatial positions.

Both variants return raw logits: ``nn.CrossEntropyLoss`` applies ``log_softmax``
itself, so a softmax in ``forward`` would apply it twice and cap the attainable
loss well above zero.
"""


from __future__ import annotations

import torch
from torch import nn


class SpectralAttention(nn.Module):
    """CBAM channel attention over the feature-map channels of a 3D tensor."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        avg = self.mlp(x.mean(dim=(2, 3, 4)))
        mx = self.mlp(x.amax(dim=(2, 3, 4)))
        weight = torch.sigmoid(avg + mx).view(b, c, 1, 1, 1)
        return x * weight


class SpatialAttention(nn.Module):
    """CBAM spatial attention, pooling over channel and spectral axes."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(
            2, 1, kernel_size=(1, kernel_size, kernel_size), padding=(0, padding, padding)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        weight = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * weight


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 8, kernel_size: int = 7):
        super().__init__()
        self.channel = SpectralAttention(channels, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.spatial(self.channel(x))


class HybridSN(nn.Module):
    """3D convolutions for spectral-spatial features, then a 2D convolution.

    Set ``use_cbam`` to insert a CBAM block after each 3D stage.
    """

    def __init__(
        self,
        n_classes: int,
        window: int = 25,
        n_bands: int = 15,
        use_cbam: bool = False,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.use_cbam = use_cbam

        def stage(in_ch: int, out_ch: int, spectral: int) -> nn.Module:
            layers: list[nn.Module] = [
                nn.Conv3d(in_ch, out_ch, kernel_size=(spectral, 3, 3)),
                nn.ReLU(inplace=True),
            ]
            if use_cbam:
                layers.append(CBAM(out_ch))
            return nn.Sequential(*layers)

        self.block1 = stage(1, 8, 7)
        self.block2 = stage(8, 16, 5)
        self.block3 = stage(16, 32, 3)

        spectral_out = n_bands - 6 - 4 - 2
        spatial_out = window - 6
        if spectral_out < 1 or spatial_out < 3:
            raise ValueError(f"window={window}, n_bands={n_bands} too small for the 3D stack")
        self.conv2d = nn.Conv2d(32 * spectral_out, 64, kernel_size=3)
        flat = 64 * (spatial_out - 2) ** 2
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, bands, H, W) -> logits (B, n_classes). No softmax here."""
        x = self.block3(self.block2(self.block1(x)))
        b, c, d, h, w = x.shape
        x = x.reshape(b, c * d, h, w)
        x = torch.relu(self.conv2d(x))
        return self.head(x)
