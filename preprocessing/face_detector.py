"""Face detection with MTCNN (primary) and RetinaFace (fallback)."""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class FaceDetector:
    """Dual-backend face detector: MTCNN primary, RetinaFace fallback."""

    def __init__(self, backend: str = "mtcnn", margin: int = 40, min_face_size: int = 60, device: str = "cuda"):
        self.margin = margin
        self.min_face_size = min_face_size
        self.device = device
        self._mtcnn = None
        self._retina = None

        if backend == "mtcnn":
            self._init_mtcnn()
        else:
            self._init_retinaface()

    def _init_mtcnn(self):
        from facenet_pytorch import MTCNN
        self._mtcnn = MTCNN(
            image_size=224,
            margin=self.margin,
            min_face_size=self.min_face_size,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            keep_all=False,
            device=self.device,
            post_process=False,
        )

    def _init_retinaface(self):
        from insightface.app import FaceAnalysis
        self._retina = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._retina.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=(640, 640))

    def detect(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Detect and return the largest face crop and landmarks.

        Returns:
            (face_crop, landmarks) or None if no face found.
        """
        result = self._detect_mtcnn(frame) if self._mtcnn else None
        if result is None and self._mtcnn is None:
            # Only try RetinaFace if MTCNN wasn't the primary backend
            try:
                if self._retina is None:
                    self._init_retinaface()
                result = self._detect_retinaface(frame)
            except (ImportError, Exception):
                pass
        return result

    def _detect_mtcnn(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        from PIL import Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        boxes, probs, landmarks = self._mtcnn.detect(pil_img, landmarks=True)
        if boxes is None or len(boxes) == 0:
            return None
        # Pick highest probability
        idx = np.argmax(probs)
        box = boxes[idx].astype(int)
        lm = landmarks[idx] if landmarks is not None else np.array([])
        face = self._crop_face(frame, box)
        if face is None:
            return None
        return face, lm

    def _detect_retinaface(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        faces = self._retina.get(frame)
        if not faces:
            return None
        # Pick largest face by bounding-box area
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        box = largest.bbox.astype(int)
        lm = largest.landmark_2d_106 if hasattr(largest, "landmark_2d_106") else largest.kps
        face = self._crop_face(frame, box)
        if face is None:
            return None
        return face, lm

    def _crop_face(self, frame: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1 = max(0, box[0] - self.margin)
        y1 = max(0, box[1] - self.margin)
        x2 = min(w, box[2] + self.margin)
        y2 = min(h, box[3] + self.margin)
        if (x2 - x1) < self.min_face_size or (y2 - y1) < self.min_face_size:
            return None
        return frame[y1:y2, x1:x2]
