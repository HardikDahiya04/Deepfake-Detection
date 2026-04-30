"""Dataset classes for deepfake detection training and evaluation."""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from preprocessing.face_detector import FaceDetector
from preprocessing.face_aligner import FaceAligner
from preprocessing.frame_sampler import FrameSampler
from preprocessing.augmentation import get_train_transforms, get_val_transforms

logger = logging.getLogger("deepfake")


class DeepfakeVideoDataset(Dataset):
    """Dataset that loads videos, extracts faces, and returns frame sequences.

    Expected directory structure:
        data_root/
            real/
                video1.mp4
                video2.mp4
            fake/
                video1.mp4
                video2.mp4
    """

    def __init__(
        self,
        data_root: str,
        num_frames: int = 16,
        frame_size: int = 224,
        is_training: bool = True,
        face_backend: str = "mtcnn",
        device: str = "cpu",
        cache_dir: Optional[str] = None,
    ):
        self.data_root = Path(data_root)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.is_training = is_training
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Build file list
        self.samples: List[Tuple[str, int]] = []
        self._scan_directory()

        # Preprocessing components
        self.sampler = FrameSampler(
            num_frames=num_frames,
            strategy="uniform_random" if is_training else "uniform",
        )
        self.detector = FaceDetector(backend=face_backend, margin=40, device=device)
        self.aligner = FaceAligner(output_size=frame_size)
        self.transform = (
            get_train_transforms((frame_size, frame_size))
            if is_training
            else get_val_transforms((frame_size, frame_size))
        )

        logger.info(f"Dataset: {len(self.samples)} videos from {data_root} (train={is_training})")

    def _scan_directory(self):
        """Scan data_root for real/ and fake/ subdirectories."""
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        for label_name, label_id in [("real", 0), ("fake", 1)]:
            label_dir = self.data_root / label_name
            if not label_dir.exists():
                logger.warning(f"Directory not found: {label_dir}")
                continue
            for f in sorted(label_dir.iterdir()):
                if f.suffix.lower() in video_exts:
                    self.samples.append((str(f), label_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, label = self.samples[idx]

        # Check cache
        cached = self._load_cache(video_path)
        if cached is not None:
            return cached, label

        # Process video
        tensor = self._process_video(video_path)
        if tensor is None:
            # Return a zero tensor as fallback
            tensor = torch.zeros(self.num_frames, 3, self.frame_size, self.frame_size)

        self._save_cache(video_path, tensor)
        return tensor, label

    def _process_video(self, video_path: str) -> Optional[torch.Tensor]:
        """Extract, detect faces, align, and transform frames."""
        try:
            frames = self.sampler.sample(video_path)
        except ValueError as e:
            logger.warning(f"Frame sampling failed for {video_path}: {e}")
            return None

        processed = []
        for frame in frames:
            result = self.detector.detect(frame)
            if result is not None:
                face, landmarks = result
                face = self.aligner.align(face, landmarks)
            else:
                # Center crop fallback
                h, w = frame.shape[:2]
                s = min(h, w)
                y, x = (h - s) // 2, (w - s) // 2
                face = cv2.resize(frame[y:y + s, x:x + s], (self.frame_size, self.frame_size))

            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            processed.append(self.transform(face_rgb))

        if not processed:
            return None

        # Pad to num_frames
        while len(processed) < self.num_frames:
            processed.append(processed[-1])
        processed = processed[:self.num_frames]

        return torch.stack(processed)

    def _cache_path(self, video_path: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        p = Path(video_path)
        label_subdir = p.parent.name  # "real" or "fake"
        return self.cache_dir / label_subdir / f"{p.stem}.pt"

    def _load_cache(self, video_path: str) -> Optional[torch.Tensor]:
        cp = self._cache_path(video_path)
        if cp and cp.exists():
            return torch.load(cp, weights_only=True)
        return None

    def _save_cache(self, video_path: str, tensor: torch.Tensor):
        cp = self._cache_path(video_path)
        if cp:
            cp.parent.mkdir(parents=True, exist_ok=True)
            torch.save(tensor, cp)


def create_dataloaders(
    train_root: str,
    val_root: str,
    cfg=None,
    test_root: str = None,
) -> Dict[str, DataLoader]:
    """Create train, validation, and optional test dataloaders."""
    from utils.config import Config
    if cfg is None:
        cfg = Config()

    loaders = {}

    train_ds = DeepfakeVideoDataset(
        data_root=train_root,
        num_frames=cfg.data.num_frames,
        frame_size=cfg.data.frame_size[0],
        is_training=True,
        face_backend=cfg.data.face_detection_backend,
        device="cpu",  # detector on CPU for dataloader workers
        cache_dir="data_cache/train",
    )
    loaders["train"] = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        drop_last=True,
    )

    val_ds = DeepfakeVideoDataset(
        data_root=val_root,
        num_frames=cfg.data.num_frames,
        frame_size=cfg.data.frame_size[0],
        is_training=False,
        face_backend=cfg.data.face_detection_backend,
        device="cpu",
        cache_dir="data_cache/val",
    )
    loaders["val"] = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
    )

    if test_root:
        test_ds = DeepfakeVideoDataset(
            data_root=test_root,
            num_frames=cfg.data.num_frames,
            frame_size=cfg.data.frame_size[0],
            is_training=False,
            face_backend=cfg.data.face_detection_backend,
            device="cpu",
        )
        loaders["test"] = DataLoader(
            test_ds,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.train.num_workers,
            pin_memory=cfg.train.pin_memory,
        )

    return loaders
