"""Visualization utilities for model evaluation and explainability."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from sklearn.metrics import roc_curve, auc


def plot_roc_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[str] = None,
):
    """Plot and optionally save the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_training_history(history: dict, save_path: Optional[str] = None):
    """Plot training loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc")
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_attention_heatmap(
    frame: np.ndarray,
    attention_map: np.ndarray,
    title: str = "Attention Heatmap",
    save_path: Optional[str] = None,
):
    """Overlay CBAM attention heatmap on a frame for explainability."""
    import cv2

    # Resize attention map to frame size
    h, w = frame.shape[:2]
    attn_resized = cv2.resize(attention_map, (w, h))
    attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)

    # Create heatmap
    heatmap = cv2.applyColorMap((attn_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay
    if frame.dtype != np.uint8:
        frame = (frame * 255).astype(np.uint8)
    overlay = cv2.addWeighted(frame, 0.6, heatmap_rgb, 0.4, 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(frame)
    axes[0].set_title("Original Frame")
    axes[0].axis("off")

    axes[1].imshow(attn_norm, cmap="jet")
    axes[1].set_title("Attention Map")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
