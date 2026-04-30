"""Data augmentation transforms for training."""

import cv2
import numpy as np
import torchvision.transforms as T
from typing import Tuple


def get_train_transforms(
    size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> T.Compose:
    """Training augmentation pipeline."""
    return T.Compose([
        T.ToPILImage(),
        T.Resize(size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
        T.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])


def get_val_transforms(
    size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> T.Compose:
    """Validation / inference transforms (no augmentation)."""
    return T.Compose([
        T.ToPILImage(),
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def add_gaussian_noise(image: np.ndarray, var: float = 0.01) -> np.ndarray:
    """Add Gaussian noise to an image."""
    noise = np.random.normal(0, var ** 0.5, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) / 255.0 + noise, 0, 1)
    return (noisy * 255).astype(np.uint8)
