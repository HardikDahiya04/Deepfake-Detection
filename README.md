# Deepfake Short Video Detection using Spatial-Temporal Attention Models

**B.Tech CSE Project — Vellore Institute of Technology, Chennai**

Spatial-Temporal Attention Model for detecting deepfake videos. Combines **ResNeXt-50** (spatial features), **CBAM** (attention), and **BiLSTM** (temporal modeling) for binary classification: **REAL** vs **FAKE**.

**Authors:** Shaurya Naithani (22BCE1461), Yash Srivastava (22BCE1968), Hardik Dahiya (22BCE1301)
**Guide:** Dr. S. Geetha, Associate Professor, SCOPE, VIT Chennai

## Architecture

```
Video → Frame Sampling → Face Detection (MTCNN/RetinaFace) → Face Alignment
     → ResNeXt-50 (per-frame features) → CBAM Attention
     → BiLSTM (temporal modeling) → Classification (Real/Fake)
```

## Project Structure

```
├── preprocessing/        # Face detection, alignment, frame sampling, augmentation
├── models/               # CBAM attention, CNN backbone, BiLSTM temporal model
├── training/             # Trainer with phased strategy, focal loss
├── evaluation/           # Metrics (Accuracy, F1, AUC, EER), visualization
├── utils/                # Config, helpers, checkpointing
├── app/                  # FastAPI backend
├── frontend/             # Web UI
├── model.py              # Full DeepfakeDetector model
├── dataset.py            # Video dataset loader
├── train.py              # Training entry point
├── test.py               # Evaluation entry point
└── infer.py              # Single-video inference
```

## Setup

```bash
pip install -r requirements.txt
```

### Dataset Preparation

Organize videos into `real/` and `fake/` subdirectories:

```
data/
├── train/
│   ├── real/    # Real video files (.mp4, .avi, etc.)
│   └── fake/    # Deepfake video files
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

**Supported datasets:** FaceForensics++, Celeb-DFv2, DFDC

## Training

```bash
# Two-phase training: Phase 1 (frozen backbone) → Phase 2 (full fine-tuning)
python train.py \
    --train-dir data/train \
    --val-dir data/val \
    --epochs 30 \
    --batch-size 8 \
    --lr 1e-4 \
    --num-frames 16
```

## Evaluation

```bash
python test.py \
    --test-dir data/test \
    --checkpoint checkpoints/best_model.pt \
    --dataset-name "FaceForensics++"
```

Outputs: Accuracy, F1, AUC, EER, Log Loss, ROC curve, confusion matrix.

## Inference

```bash
python infer.py \
    --video path/to/video.mp4 \
    --checkpoint checkpoints/best_model.pt \
    --visualize
```

Output:
```json
{
  "prediction": "FAKE",
  "confidence": 0.9234,
  "fake_probability": 0.9234
}

```

## API Server

```bash
# Set checkpoint path
export MODEL_CHECKPOINT=checkpoints/best_model.pt

# Start server
uvicorn app.api:app --host 0.0.0.0 --port 8000

# Test
curl -X POST http://localhost:8000/predict -F "video=@test_video.mp4"
```

## React Frontend

```bash
cd frontend
npm install
npm start
```

Opens at `http://localhost:3000`. Upload a video and get real-time deepfake prediction with confidence score.

## Training Strategy

| Phase | Backbone | CBAM + BiLSTM | Description |
|-------|----------|---------------|-------------|
| 1     | Frozen   | Training      | Learn attention & temporal patterns |
| 2     | Fine-tune (0.1x LR) | Training | End-to-end optimization |

## Tech Stack

- **PyTorch** + **timm** (ResNeXt-50 pretrained backbone)
- **facenet-pytorch** (MTCNN face detection)
- **insightface** (RetinaFace fallback)
- **FastAPI** (REST API)
- **React** (Frontend UI)
- Mixed precision training (AMP)
- Focal Loss for class imbalance

## Expected Performance

| Evaluation Scenario | Accuracy | AUC | F1 Score |
|---|---|---|---|
| FaceForensics++ (intra-dataset) | >90% | >0.95 | >0.90 |
| FF++ → Celeb-DFv2 (cross-dataset) | 80–88% | 0.85–0.93 | 0.78–0.87 |
| FF++ → DFDC (stress test) | 60–72% | 0.65–0.76 | 0.58–0.70 |

## Dataset Strategy

| Dataset | Role | Size |
|---|---|---|
| FaceForensics++ | Training | ~5,000 videos |
| Celeb-DFv2 | Validation / Generalization | ~6,000 videos |
| DFDC | Stress Test | ~128,000 videos |
