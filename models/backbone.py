"""CNN backbone for spatial feature extraction."""

import torch
import torch.nn as nn
import timm

from models.cbam import CBAM


class SpatialBackbone(nn.Module):
    """ResNeXt-50 backbone with CBAM attention for frame-level features."""

    def __init__(
        self,
        model_name: str = "resnext50_32x4d",
        pretrained: bool = True,
        feature_dim: int = 2048,
        cbam_reduction: int = 16,
        cbam_kernel_size: int = 7,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        # Load pretrained backbone, remove classification head
        base = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="")
        self.features = base

        # CBAM attention on the final feature map
        self.cbam = CBAM(feature_dim, reduction=cbam_reduction, kernel_size=cbam_kernel_size)

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial features from a single frame.

        Args:
            x: (B, C, H, W) input frame tensor.

        Returns:
            (B, feature_dim) feature vector.
        """
        feat = self.features(x)         # (B, feature_dim, h, w)
        feat = self.cbam(feat)           # apply attention
        feat = self.pool(feat)           # (B, feature_dim, 1, 1)
        return feat.flatten(1)           # (B, feature_dim)

    def freeze(self):
        """Freeze the CNN backbone parameters (Phase 1 training)."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters (Phase 2 fine-tuning)."""
        for param in self.features.parameters():
            param.requires_grad = True
