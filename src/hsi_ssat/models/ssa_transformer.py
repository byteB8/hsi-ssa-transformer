"""SSA-Transformer (Dang et al., 2022).

Spectral-Spatial Attention Transformer with Dense Connection, CIN 2022,
doi:10.1155/2022/7071485.

Pipeline, following the paper's equations (1)-(8):

    y'    = Conv1(y)              two 3x3 conv layers, spectral depth preserved
    y''   = Mse(y') * y'          spectral (channel) attention, eq. 2-3
    y'''  = Msa(y'') * y'' + y    spatial attention with a residual to the input
    y'''' = Conv2(y''')           one 1x1 conv, reducing C -> k

The feature map is then cut into non-overlapping p x p patches (p = 3), each
flattened to a p*p*k vector, given a class token and learned position codes, and
passed through transformer encoder blocks wired with dense connections: block i
receives the projected concatenation of the outputs of all preceding blocks.

Two details are not stated in the paper and are exposed as configuration here:
the number of encoder blocks and the retained band count k. Defaults are 4 and
32.
"""

from __future__ import annotations

import torch
from torch import nn


class SpectralAttentionModule(nn.Module):
    """SeAM, eq. 2-3. Shared MLP over global average- and max-pooled bands.

    The paper applies ReLU between the two fully connected layers and again to
    the summed vector before the sigmoid.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        avg = self.fc2(torch.relu(self.fc1(x.mean(dim=(2, 3)))))
        mx = self.fc2(torch.relu(self.fc1(x.amax(dim=(2, 3)))))
        weight = torch.sigmoid(torch.relu(avg + mx)).view(b, c, 1, 1)
        return x * weight


class SpatialAttentionModule(nn.Module):
    """SaAM, eq. 4-5. Average and max over the spectral axis, then a conv."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        weight = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * weight


class SpectralSpatialAttention(nn.Module):
    """The full CNN front end, eq. 1. Conv1 -> SeAM -> SaAM (+input) -> Conv2."""

    def __init__(self, in_channels: int, out_channels: int, reduction: int = 8):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.spectral = SpectralAttentionModule(in_channels, reduction)
        self.spatial = SpatialAttentionModule()
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.conv1(x)
        y = self.spectral(y)
        y = self.spatial(y) + residual
        return self.conv2(y)


class EncoderBlock(nn.Module):
    """Pre-norm transformer encoder block with a GeLU feed-forward net, eq. 6-8."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attended, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attended
        return x + self.ffn(self.norm2(x))


class SSATransformer(nn.Module):
    def __init__(
        self,
        n_classes: int,
        n_bands: int,
        window: int = 15,
        patch: int = 3,
        retained_bands: int = 32,
        depth: int = 4,
        heads: int = 6,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        reduction: int = 8,
        dense: bool = True,
    ):
        super().__init__()
        if window % patch:
            raise ValueError(f"window {window} must be divisible by patch {patch}")
        self.patch = patch
        self.dense = dense
        self.depth = depth

        self.front_end = SpectralSpatialAttention(n_bands, retained_bands, reduction)

        n_tokens = (window // patch) ** 2
        dim = retained_bands * patch * patch
        if dim % heads:
            raise ValueError(f"embed dim {dim} not divisible by heads {heads}")

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position = nn.Parameter(torch.zeros(1, n_tokens + 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            EncoderBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)
        )
        # Dense connection: block i consumes every earlier block's output, so the
        # concatenation is projected back to `dim` before being fed forward.
        # Block i consumes the outputs of the i blocks before it.
        self.fusions = nn.ModuleList(
            nn.Linear(dim * i, dim) if dense and i > 0 else nn.Identity() for i in range(depth)
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

    def tokenise(self, x: torch.Tensor) -> torch.Tensor:
        """(B, k, H, W) -> (B, N, k*p*p) by cutting non-overlapping p x p patches."""
        b, k, h, w = x.shape
        p = self.patch
        x = x.reshape(b, k, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(b, (h // p) * (w // p), k * p * p)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, bands, H, W) -> logits (B, n_classes). No softmax here."""
        x = self.front_end(x)
        tokens = self.tokenise(x)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.position
        tokens = self.dropout(tokens)

        outputs: list[torch.Tensor] = []
        current = tokens
        for index, block in enumerate(self.blocks):
            if self.dense and index > 0:
                current = self.fusions[index](torch.cat(outputs, dim=-1))
            current = block(current)
            outputs.append(current)

        return self.head(self.norm(current)[:, 0])
