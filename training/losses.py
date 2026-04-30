"""Loss functions for deepfake detection training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class CombinedLoss(nn.Module):
    """CrossEntropy + Focal Loss combination."""

    def __init__(self, focal_alpha: float = 0.25, focal_gamma: float = 2.0, focal_weight: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.focal_weight = focal_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(inputs, targets)
        focal_loss = self.focal(inputs, targets)
        return (1 - self.focal_weight) * ce_loss + self.focal_weight * focal_loss
