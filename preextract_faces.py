"""
Pre-extract face crops from all videos and save as .pt tensors.
Run this ONCE before training — eliminates face detection bottleneck during training.
"""

import os
import torch
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from preprocessing.pipeline import PreprocessingPipeline
from utils.config import Config

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SPLITS = ["train", "val", "test"]
LABELS = ["real", "fake"]
DATA_ROOT = Path("data")
CACHE_ROOT = Path("data_cache")
NUM_FRAMES = 16
NUM_WORKERS = 4   # parallel video processing on CPU


def process_video(video_path: Path, out_path: Path, pipeline: PreprocessingPipeline) -> str:
    if out_path.exists():
        return f"SKIP {video_path.name}"
    try:
        tensor = pipeline.process_video(str(video_path))
        if tensor is None:
            return f"FAIL {video_path.name} (no face detected)"
        torch.save(tensor, out_path)
        return f"OK   {video_path.name} → shape {tuple(tensor.shape)}"
    except Exception as e:
        return f"ERR  {video_path.name}: {e}"


def main():
    cfg = Config()
    cfg.data.num_frames = NUM_FRAMES

    # Use CPU pipeline for preprocessing (MTCNN runs on CPU anyway)
    pipeline = PreprocessingPipeline(num_frames=NUM_FRAMES, is_training=False, device="cpu")

    total, done, failed = 0, 0, 0

    for split in SPLITS:
        for label in LABELS:
            src_dir = DATA_ROOT / split / label
            dst_dir = CACHE_ROOT / split / label
            if not src_dir.exists():
                continue
            dst_dir.mkdir(parents=True, exist_ok=True)

            videos = sorted(src_dir.glob("*.mp4"))
            total += len(videos)
            logger.info(f"\n── {split}/{label}: {len(videos)} videos ──")

            tasks = [(v, dst_dir / (v.stem + ".pt")) for v in videos]

            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = {
                    executor.submit(process_video, v, o, pipeline): v
                    for v, o in tasks
                }
                for future in as_completed(futures):
                    msg = future.result()
                    if msg.startswith("OK"):
                        done += 1
                    elif msg.startswith("FAIL") or msg.startswith("ERR"):
                        failed += 1
                        logger.warning(msg)
                    else:
                        done += 1  # skipped = already done
                    logger.info(msg)

    logger.info(f"\n{'='*50}")
    logger.info(f"Done: {done}/{total} | Failed/skipped: {failed}")
    logger.info(f"Cache saved to: {CACHE_ROOT.resolve()}")
    logger.info("Now run training with: python3 train.py --use-cache")


if __name__ == "__main__":
    main()
