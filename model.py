"""DeepfakeDetector: full spatial-temporal attention model."""

import torch
import torch.nn as nn

from models.backbone import SpatialBackbone
from models.temporal import TemporalModel
from utils.config import ModelConfig


class DeepfakeDetector(nn.Module):
    """End-to-end deepfake detection: ResNeXt + CBAM + BiLSTM + classifier."""

    def __init__(self, cfg: ModelConfig = None):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()

        # Spatial feature extractor (per-frame)
        self.spatial = SpatialBackbone(
            model_name=cfg.backbone,
            pretrained=cfg.backbone_pretrained,
            feature_dim=cfg.feature_dim,
            cbam_reduction=cfg.cbam_reduction,
            cbam_kernel_size=cfg.cbam_kernel_size,
        )

        # Temporal model (across frames)
        self.temporal = TemporalModel(
            input_dim=cfg.feature_dim,
            hidden_dim=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            dropout=cfg.lstm_dropout,
            bidirectional=cfg.bidirectional,
        )

        # Classification head
        temporal_out = cfg.lstm_hidden * (2 if cfg.bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(temporal_out, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout * 0.5),
            nn.Linear(256, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, C, H, W) batch of video frame sequences.

        Returns:
            (B, num_classes) classification logits.
        """
        B, T, C, H, W = x.shape

        # Extract per-frame spatial features
        frames = x.reshape(B * T, C, H, W)
        features = self.spatial(frames)         # (B*T, feat_dim)
        features = features.reshape(B, T, -1)   # (B, T, feat_dim)

        # Temporal modeling
        temporal_out = self.temporal(features)   # (B, temporal_dim)

        # Classify
        logits = self.classifier(temporal_out)   # (B, num_classes)
        return logits

    def freeze_backbone(self):
        """Phase 1: freeze CNN, train only CBAM + BiLSTM + classifier."""
        self.spatial.freeze()

    def unfreeze_all(self):
        """Phase 2: unfreeze everything for end-to-end fine-tuning."""
        self.spatial.unfreeze()

    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CBAM spatial attention maps for visualization.

        Args:
            x: (B, T, C, H, W) input.

        Returns:
            (B, T, H', W') attention heatmaps.
        """
        B, T, C, H, W = x.shape
        frames = x.reshape(B * T, C, H, W)

        # Forward through backbone features
        feat = self.spatial.features(frames)
        # Get channel attention
        ca_feat = self.spatial.cbam.channel_attn(feat)
        # Get spatial attention map
        avg = ca_feat.mean(dim=1, keepdim=True)
        mx = ca_feat.amax(dim=1, keepdim=True)
        cat = torch.cat([avg, mx], dim=1)
        attn_map = torch.sigmoid(self.spatial.cbam.spatial_attn.conv(cat))
        attn_map = attn_map.squeeze(1)  # (B*T, H', W')

        return attn_map.reshape(B, T, attn_map.shape[-2], attn_map.shape[-1])
