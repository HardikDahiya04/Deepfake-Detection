"""Frame sampling strategies for video preprocessing."""

import cv2
import numpy as np
from typing import List


class FrameSampler:
    """Sample frames from video using uniform + random strategies."""

    def __init__(self, num_frames: int = 16, strategy: str = "uniform_random"):
        self.num_frames = num_frames
        self.strategy = strategy

    def sample(self, video_path: str) -> List[np.ndarray]:
        """Extract sampled frames from a video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Empty video: {video_path}")

        indices = self._get_indices(total_frames)
        frames = []
        for idx in sorted(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames extracted from: {video_path}")
        return frames

    def _get_indices(self, total: int) -> List[int]:
        n = min(self.num_frames, total)
        if self.strategy == "uniform":
            return np.linspace(0, total - 1, n, dtype=int).tolist()
        elif self.strategy == "random":
            return sorted(np.random.choice(total, n, replace=False).tolist())
        else:  # uniform_random: uniform base + jitter
            base = np.linspace(0, total - 1, n, dtype=float)
            step = (total - 1) / max(n - 1, 1)
            jitter = np.random.uniform(-step * 0.3, step * 0.3, n)
            indices = np.clip(base + jitter, 0, total - 1).astype(int)
            return sorted(set(indices.tolist()))[:n]
