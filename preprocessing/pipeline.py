"""End-to-end preprocessing pipeline: video → face-cropped frame tensors."""

import cv2
import numpy as np
import torch
from typing import Optional

from preprocessing.face_detector import FaceDetector
from preprocessing.face_aligner import FaceAligner
from preprocessing.frame_sampler import FrameSampler
from preprocessing.augmentation import get_train_transforms, get_val_transforms


class PreprocessingPipeline:
    """Full pipeline: video → sampled face-cropped tensor sequence."""

    def __init__(
        self,
        num_frames: int = 16,
        frame_size: int = 224,
        face_backend: str = "mtcnn",
        margin: int = 40,
        is_training: bool = True,
        device: str = "cuda",
    ):
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.sampler = FrameSampler(num_frames=num_frames, strategy="uniform_random" if is_training else "uniform")
        self.detector = FaceDetector(backend=face_backend, margin=margin, device=device)
        self.aligner = FaceAligner(output_size=frame_size)
        self.transform = get_train_transforms((frame_size, frame_size)) if is_training else get_val_transforms((frame_size, frame_size))

    def process_video(self, video_path: str) -> Optional[torch.Tensor]:
        """Process a video file into a tensor of shape (T, C, H, W).

        Returns None if the video cannot be processed.
        """
        try:
            frames = self.sampler.sample(video_path)
        except ValueError:
            return None

        processed = []
        for frame in frames:
            result = self.detector.detect(frame)
            if result is not None:
                face, landmarks = result
                face = self.aligner.align(face, landmarks)
            else:
                # Fallback: center-crop the frame
                face = self._center_crop(frame)

            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            tensor = self.transform(face_rgb)
            processed.append(tensor)

        if len(processed) == 0:
            return None

        # Pad or truncate to num_frames
        while len(processed) < self.num_frames:
            processed.append(processed[-1])
        processed = processed[:self.num_frames]

        return torch.stack(processed)  # (T, C, H, W)

    def _center_crop(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        size = min(h, w)
        y = (h - size) // 2
        x = (w - size) // 2
        crop = frame[y:y + size, x:x + size]
        return cv2.resize(crop, (self.frame_size, self.frame_size))
