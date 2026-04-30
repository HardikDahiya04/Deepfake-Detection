# Deepfake Video Detection

A deep learning system for detecting deepfake videos using a Spatial-Temporal Attention model. Built as a B.Tech CSE final year project at **Vellore Institute of Technology, Chennai**.

**Authors:** Shaurya Naithani (22BCE1461), Yash Srivastava (22BCE1968), Hardik Dahiya (22BCE1301)  
**Guide:** Dr. S. Geetha, Associate Professor, SCOPE, VIT Chennai

---

## How It Works

```
Video → Frame Sampling → Face Detection (MTCNN/RetinaFace) → Face Alignment
     → ResNeXt-50 (per-frame features) → CBAM Attention
     → BiLSTM (temporal modeling) → Real / Fake
```

The model combines:
- **ResNeXt-50** — extracts spatial features from each frame
- **CBAM** — channel & spatial attention to focus on manipulated regions
- **BiLSTM** — models temporal inconsistencies across frames

---

## Project Structure

```
├── model.py              # Full DeepfakeDetector model
├── dataset.py            # Video dataset loader
├── train.py              # Training entry point
├── test.py               # Evaluation entry point
├── infer.py              # Single-video inference
├── prepare_dataset.py    # Dataset preparation script
├── preextract_faces.py   # Pre-extract faces for faster training
├── preprocessing/        # Face detection, alignment, augmentation
├── models/               # CBAM, CNN backbone, BiLSTM modules
├── training/             # Trainer, focal loss, phased strategy
├── evaluation/           # Metrics (Accuracy, F1, AUC, EER), plots
├── utils/                # Config, helpers, checkpointing
├── app/                  # FastAPI backend
└── frontend/             # React web UI
```

---

## Installation

### Requirements
- Python 3.9+
- CUDA-capable GPU (recommended) or CPU
- Node.js 16+ (only for the frontend)

### 1. Clone the repo

```bash
git clone https://github.com/HardikDahiya04/Deepfake-Detection.git
cd Deepfake-Detection
```

### 2. Create a virtual environment

```bash
python -m venv venv

# On Linux/macOS
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

> If you don't have a GPU, PyTorch will fall back to CPU automatically. Training will be much slower but inference still works fine.

---

## Dataset Setup

Organize your videos into this folder structure before training:

```
data/
├── train/
│   ├── real/     # Real video files (.mp4, .avi, etc.)
│   └── fake/     # Deepfake video files
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

**Supported public datasets:** FaceForensics++, Celeb-DFv2, DFDC

Once organized, optionally pre-extract faces for faster training:

```bash
python preextract_faces.py --data-dir data/ --output-dir data_cache/
```

---

## Training

```bash
python train.py \
    --train-dir data/train \
    --val-dir data/val \
    --epochs 30 \
    --batch-size 8 \
    --lr 1e-4 \
    --num-frames 16
```

Training uses a **two-phase strategy**:
- **Phase 1:** Backbone frozen — trains only CBAM + BiLSTM layers
- **Phase 2:** Full fine-tuning with a 10x lower learning rate on the backbone

Checkpoints are saved to `checkpoints/` after each epoch. The best model is saved as `checkpoints/best_model.pt`.

To resume a stopped training run:

```bash
python train.py --train-dir data/train --val-dir data/val --resume checkpoints/best_model.pt
```

---

## Evaluation

```bash
python test.py \
    --test-dir data/test \
    --checkpoint checkpoints/best_model.pt \
    --dataset-name "FaceForensics++"
```

Outputs: Accuracy, F1, AUC, EER, Log Loss, ROC curve, and confusion matrix.

### Expected performance

| Scenario | Accuracy | AUC | F1 |
|---|---|---|---|
| FaceForensics++ (intra-dataset) | >90% | >0.95 | >0.90 |
| FF++ → Celeb-DFv2 (cross-dataset) | 80–88% | 0.85–0.93 | 0.78–0.87 |
| FF++ → DFDC (stress test) | 60–72% | 0.65–0.76 | 0.58–0.70 |

---

## Inference (Single Video)

```bash
python infer.py \
    --video path/to/video.mp4 \
    --checkpoint checkpoints/best_model.pt \
    --visualize
```

Example output:

```json
{
  "prediction": "FAKE",
  "confidence": 0.9234,
  "fake_probability": 0.9234,
  "processing_details": {
    "frames_analyzed": 16,
    "video_duration_seconds": 8.4,
    "processing_time_seconds": 1.23,
    "device": "cuda"
  }
}
```

Use `--visualize` to save attention heatmap overlays to `outputs/inference/`.

---

## Running the Web App

The project includes a FastAPI backend and a React frontend for a browser-based interface.

### Backend (API server)

```bash
export MODEL_CHECKPOINT=checkpoints/best_model.pt
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Test it:

```bash
curl -X POST http://localhost:8000/predict -F "video=@test_video.mp4"
```

### Frontend (React UI)

```bash
cd frontend
npm install
npm start
```

Opens at `http://localhost:3000`. Upload a video and get a real-time prediction with confidence score and frame-by-frame attention visualization.

---

## Tech Stack

| Component | Library |
|---|---|
| Model training | PyTorch, timm |
| Face detection | facenet-pytorch (MTCNN), insightface (RetinaFace) |
| Video processing | OpenCV, albumentations |
| API | FastAPI, uvicorn |
| Frontend | React |
| Metrics | scikit-learn, matplotlib, seaborn |

---

## Common Issues

**`No faces detected` warning during training**  
Some videos may have no detectable faces. These are skipped automatically. If most videos are skipped, check that your video files are not corrupted.

**CUDA out of memory**  
Reduce `--batch-size` to 4 or lower, or reduce `--num-frames` to 8.

**Slow training on CPU**  
Set `--workers 0` (default) to avoid multiprocessing issues on Windows/WSL. Training on CPU with 30 epochs can take several hours — a GPU is strongly recommended.

**InsightFace install fails**  
Try: `pip install insightface --no-build-isolation`
