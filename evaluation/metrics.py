"""Evaluation metrics for deepfake detection."""

import numpy as np
from typing import Dict, Tuple

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    log_loss,
    classification_report,
    confusion_matrix,
)


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """Compute Equal Error Rate (EER).

    Returns:
        (eer, threshold) at the EER point.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return float(eer), float(thresholds[idx])


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
) -> Dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Predicted labels (0 or 1).
        y_probs: Predicted probabilities for the positive class.

    Returns:
        Dictionary of metric name → value.
    """
    metrics = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["f1_score"] = float(f1_score(y_true, y_pred, average="binary"))
    metrics["auc"] = float(roc_auc_score(y_true, y_probs))
    metrics["log_loss"] = float(log_loss(y_true, y_probs))

    eer, eer_thresh = compute_eer(y_true, y_probs)
    metrics["eer"] = eer
    metrics["eer_threshold"] = eer_thresh

    return metrics


def print_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    dataset_name: str = "Test",
):
    """Print a full evaluation report."""
    metrics = compute_all_metrics(y_true, y_pred, y_probs)

    print(f"\n{'=' * 50}")
    print(f"  Evaluation Report: {dataset_name}")
    print(f"{'=' * 50}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  F1 Score:    {metrics['f1_score']:.4f}")
    print(f"  AUC:         {metrics['auc']:.4f}")
    print(f"  EER:         {metrics['eer']:.4f} (threshold={metrics['eer_threshold']:.4f})")
    print(f"  Log Loss:    {metrics['log_loss']:.4f}")
    print(f"{'=' * 50}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["REAL", "FAKE"]))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  {'':>8} Pred REAL  Pred FAKE")
    print(f"  Real    {cm[0][0]:>8}  {cm[0][1]:>8}")
    print(f"  Fake    {cm[1][0]:>8}  {cm[1][1]:>8}")

    return metrics
