"""Evaluation script for the Deepfake Detection System."""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import DeepfakeDetector
from dataset import DeepfakeVideoDataset
from evaluation.metrics import compute_all_metrics, print_evaluation_report
from evaluation.visualize import plot_roc_curve
from utils.config import Config
from utils.helpers import setup_logging, set_seed, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Deepfake Detector")
    parser.add_argument("--test-dir", type=str, required=True, help="Path to test data")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="outputs/eval")
    parser.add_argument("--dataset-name", type=str, default="Test")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    """Run evaluation and collect predictions."""
    model.eval()
    all_labels, all_probs, all_preds = [], [], []

    for videos, labels in loader:
        videos = videos.to(device, non_blocking=True)
        logits = model(videos)
        probs = F.softmax(logits, dim=1)[:, 1]  # P(fake)
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def main():
    args = parse_args()
    logger = setup_logging()
    set_seed(42)
    device = get_device(args.device)

    # Load model
    cfg = Config()
    cfg.data.num_frames = args.num_frames
    model = DeepfakeDetector(cfg.model)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # Dataset
    dataset = DeepfakeVideoDataset(
        data_root=args.test_dir,
        num_frames=args.num_frames,
        is_training=False,
        device="cpu",
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    logger.info(f"Test set: {len(dataset)} videos")

    # Evaluate
    y_true, y_pred, y_probs = evaluate(model, loader, device)
    metrics = print_evaluation_report(y_true, y_pred, y_probs, dataset_name=args.dataset_name)

    # Save ROC curve
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_roc_curve(y_true, y_probs, title=f"ROC - {args.dataset_name}", save_path=str(output_dir / "roc_curve.png"))
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
