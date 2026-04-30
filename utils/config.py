"""Central configuration for the Deepfake Detection System."""

import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataConfig:
    """Dataset and preprocessing configuration."""
    # Frame extraction
    num_frames: int = 16
    frame_size: Tuple[int, int] = (224, 224)
    # ImageNet normalization
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    # Face detection
    face_detection_backend: str = "mtcnn"  # "mtcnn" or "retinaface"
    face_margin: int = 40
    face_min_size: int = 60
    # Augmentation
    augment: bool = True
    rotation_limit: int = 10
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    gaussian_noise_var: float = 0.01


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # CNN backbone
    backbone: str = "resnext50_32x4d"
    backbone_pretrained: bool = True
    feature_dim: int = 2048
    # CBAM
    cbam_reduction: int = 16
    cbam_kernel_size: int = 7
    # BiLSTM
    lstm_hidden: int = 512
    lstm_layers: int = 2
    lstm_dropout: float = 0.3
    bidirectional: bool = True
    # Classification
    num_classes: int = 2
    dropout: float = 0.6


@dataclass
class TrainConfig:
    """Training configuration."""
    # Optimizer
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 5e-4
    momentum: float = 0.9  # for SGD
    # Scheduler
    scheduler: str = "cosine"
    warmup_epochs: int = 2
    # Training
    epochs: int = 40
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    # Loss
    focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    # Phased training
    phase1_epochs: int = 15  # freeze CNN, train CBAM + BiLSTM
    # Checkpointing
    save_dir: str = "checkpoints"
    save_best: bool = True
    early_stopping_patience: int = 8
    # Mixed precision
    use_amp: bool = True
    # Gradient clipping
    max_grad_norm: float = 1.0


@dataclass
class Config:
    """Master configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    # Paths
    data_root: str = "data"
    output_dir: str = "outputs"
    device: str = "cuda"
    seed: int = 42
