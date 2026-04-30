"""Organize Celeb-DF v2 dataset into train/val/test splits."""

import os
import shutil
import random
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR = Path("/home/hardik/Deepfake/Dataset")
OUTPUT_DIR  = Path("/home/hardik/Deepfake/data")
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10
SEED        = 42
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)

def get_videos(folder: Path) -> list:
    """Return sorted list of .mp4 files (ignore Zone.Identifier etc.)."""
    return sorted([
        f for f in folder.iterdir()
        if f.suffix.lower() == ".mp4"
    ])

def split(videos: list):
    random.shuffle(videos)
    n = len(videos)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    return videos[:n_train], videos[n_train:n_train+n_val], videos[n_train+n_val:]

def copy_videos(videos: list, dest: Path, label: str):
    dest.mkdir(parents=True, exist_ok=True)
    for v in videos:
        shutil.copy2(v, dest / v.name)
    print(f"  {label}: {len(videos)} videos → {dest}")

def main():
    # Collect real videos from both sources
    real_videos = (
        get_videos(DATASET_DIR / "Celeb-real") +
        get_videos(DATASET_DIR / "YouTube-real")
    )
    fake_videos = get_videos(DATASET_DIR / "Celeb-synthesis")

    print(f"Found {len(real_videos)} real videos, {len(fake_videos)} fake videos")

    # Split each class independently (stratified)
    real_train, real_val, real_test = split(real_videos)
    fake_train, fake_val, fake_test = split(fake_videos)

    print("\nCopying files...")
    copy_videos(real_train, OUTPUT_DIR / "train" / "real", "train/real")
    copy_videos(fake_train, OUTPUT_DIR / "train" / "fake", "train/fake")
    copy_videos(real_val,   OUTPUT_DIR / "val"   / "real", "val/real")
    copy_videos(fake_val,   OUTPUT_DIR / "val"   / "fake", "val/fake")
    copy_videos(real_test,  OUTPUT_DIR / "test"  / "real", "test/real")
    copy_videos(fake_test,  OUTPUT_DIR / "test"  / "fake", "test/fake")

    print(f"""
Dataset ready at: {OUTPUT_DIR}

Split summary:
  Train → real: {len(real_train)}, fake: {len(fake_train)}  (total: {len(real_train)+len(fake_train)})
  Val   → real: {len(real_val)},   fake: {len(fake_val)}    (total: {len(real_val)+len(fake_val)})
  Test  → real: {len(real_test)},  fake: {len(fake_test)}   (total: {len(real_test)+len(fake_test)})
""")

if __name__ == "__main__":
    main()
