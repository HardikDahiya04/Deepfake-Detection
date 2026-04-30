"""Convolutional Block Attention Module (CBAM).

Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention: learns which feature channels are important."""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid = max(in_channels // reduction, 1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        # Global average pool
        avg = x.mean(dim=[2, 3])  # (B, C)
        # Global max pool
        mx = x.amax(dim=[2, 3])  # (B, C)
        # Shared MLP
        attn = torch.sigmoid(self.shared_mlp(avg) + self.shared_mlp(mx))
        return x * attn.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    """Spatial attention: learns which spatial locations are important."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        cat = torch.cat([avg, mx], dim=1)
        attn = torch.sigmoid(self.conv(cat))
        return x * attn


class CBAM(nn.Module):
    """CBAM: sequential channel then spatial attention."""

    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x
