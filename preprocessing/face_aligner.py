"""Face alignment using landmark-based affine transformation."""

import cv2
import numpy as np
from typing import Optional

# Reference landmarks for a 224x224 aligned face (left eye, right eye, nose, mouth-left, mouth-right)
REFERENCE_LANDMARKS = np.array([
    [70.0, 112.0],   # left eye
    [154.0, 112.0],  # right eye
    [112.0, 150.0],  # nose tip
    [78.0, 180.0],   # mouth left
    [146.0, 180.0],  # mouth right
], dtype=np.float32)


class FaceAligner:
    """Align faces using landmark-based affine transforms."""

    def __init__(self, output_size: int = 224):
        self.output_size = output_size
        scale = output_size / 224.0
        self.ref_landmarks = REFERENCE_LANDMARKS * scale

    def align(self, face_img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Align a face image using detected landmarks.

        Args:
            face_img: Cropped face BGR image.
            landmarks: (N, 2) array of facial landmarks.

        Returns:
            Aligned face image of size (output_size, output_size).
        """
        if landmarks is None or len(landmarks) < 2:
            return cv2.resize(face_img, (self.output_size, self.output_size))

        src_pts = self._select_key_landmarks(landmarks)
        if src_pts is None:
            return cv2.resize(face_img, (self.output_size, self.output_size))

        dst_pts = self.ref_landmarks[:len(src_pts)]
        M = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
        if M is None:
            return cv2.resize(face_img, (self.output_size, self.output_size))

        aligned = cv2.warpAffine(face_img, M, (self.output_size, self.output_size),
                                 borderMode=cv2.BORDER_REFLECT_101)
        return aligned

    def _select_key_landmarks(self, landmarks: np.ndarray) -> Optional[np.ndarray]:
        """Extract key landmarks for alignment."""
        lm = np.array(landmarks, dtype=np.float32)
        if lm.ndim == 1:
            lm = lm.reshape(-1, 2)
        if len(lm) >= 5:
            return lm[:5]
        if len(lm) >= 2:
            return lm[:min(3, len(lm))]
        return None
