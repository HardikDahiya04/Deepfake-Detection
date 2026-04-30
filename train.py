"""Main training script for the Deepfake Detection System."""

import argparse
import sys
from pathlib import Path

import torch

from model import DeepfakeDetector
from dataset import create_dataloaders
from training.trainer import Trainer
from evaluation.visualize import plot_training_history
from utils.config import Config
from utils.helpers import setup_logging, set_seed, get_device, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(description="Train Deepfake Detector")
    parser.add_argument("--train-dir", type=str, required=True, help="Path to training data (with real/ and fake/ subdirs)")
    parser.add_argument("--val-dir", type=str, required=True, help="Path to validation data")
    parser.add_argument("--epochs", type=int, default=30, help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-frames", type=int, default=16, help="Frames per video")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers (0=main process only, safer in WSL)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup
    logger = setup_logging(log_file=f"{args.save_dir}/training.log")
    set_seed(args.seed)
    device = get_device(args.device)

    # Config
    cfg = Config()
    cfg.train.epochs = args.epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.learning_rate = args.lr
    cfg.train.save_dir = args.save_dir
    cfg.train.num_workers = args.workers
    cfg.data.num_frames = args.num_frames
    cfg.seed = args.seed

    logger.info(f"Device: {device}")
    logger.info(f"Config: epochs={cfg.train.epochs}, batch_size={cfg.train.batch_size}, "
                f"lr={cfg.train.learning_rate}, num_frames={cfg.data.num_frames}")

    # Data
    loaders = create_dataloaders(
        train_root=args.train_dir,
        val_root=args.val_dir,
        cfg=cfg,
    )
    logger.info(f"Train batches: {len(loaders['train'])}, Val batches: {len(loaders['val'])}")

    # Model
    model = DeepfakeDetector(cfg.model)
    logger.info(f"Model parameters: {count_parameters(model):,}")

    # Train
    trainer = Trainer(model, cfg, device)
    history = trainer.fit(loaders["train"], loaders["val"], resume_checkpoint=args.resume)

    # Plot training curves
    output_dir = Path(args.save_dir)
    plot_training_history(history, save_path=str(output_dir / "training_curves.png"))
    logger.info(f"Training complete. Best model saved to {args.save_dir}/best_model.pt")


if __name__ == "__main__":
    main()
