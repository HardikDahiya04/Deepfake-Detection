"""Utility helpers for the Deepfake Detection System."""

import os
import random
import logging
from pathlib import Path

import numpy as np
import torch


def setup_logging(log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Configure project-wide logging."""
    logger = logging.getLogger("deepfake")
    logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(preferred: str = "cuda") -> torch.device:
    """Return the best available device."""
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state: dict, filepath: str):
    """Save a training checkpoint."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(filepath: str, model: torch.nn.Module, optimizer=None):
    """Load a training checkpoint."""
    checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("best_metric", 0.0)


class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        else:
            self.best_score = score
            self.counter = 0
        return False
