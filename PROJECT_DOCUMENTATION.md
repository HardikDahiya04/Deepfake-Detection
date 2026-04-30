# Deepfake Detection System - Complete Technical Documentation

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Model Architecture](#2-model-architecture)
3. [Data Pipeline](#3-data-pipeline)
4. [Training Process](#4-training-process)
5. [Evaluation & Results](#5-evaluation--results)
6. [Inference Pipeline](#6-inference-pipeline)
7. [Web Application](#7-web-application)
8. [Setup & Deployment](#8-setup--deployment)

---

## 1. Project Overview

This project is a **spatial-temporal deep learning system** for detecting deepfake videos. It analyzes both the visual content of individual frames and the temporal consistency across frames to determine whether a video is real or synthetically generated.

### High-Level Flow
```
Input Video
    |
    v
[Frame Sampling] --> 16 frames extracted uniformly
    |
    v
[Face Detection] --> MTCNN detects faces in each frame
    |
    v
[Face Alignment] --> Landmark-based affine alignment to 224x224
    |
    v
[Spatial Feature Extraction] --> ResNeXt-50 + CBAM Attention per frame
    |
    v
[Temporal Modeling] --> Bidirectional LSTM across all 16 frames
    |
    v
[Classification] --> REAL or FAKE with confidence score
```

### Dataset Used
- **Celeb-DF v2** (Celebrity Deepfake Dataset)
  - `Celeb-real/` — Real celebrity videos
  - `YouTube-real/` — Real YouTube videos
  - `Celeb-synthesis/` — AI-generated deepfake videos
  - Split: 80% training, 10% validation, 10% testing

---

## 2. Model Architecture

The model (`DeepfakeDetector`) is a three-stage neural network with approximately **40.5 million parameters**.

### 2.1 Stage 1: Spatial Feature Extraction (ResNeXt-50 + CBAM)

**Backbone: ResNeXt-50-32x4d**
- A convolutional neural network pretrained on ImageNet (1.2 million images, 1000 classes).
- ResNeXt improves on ResNet by using grouped convolutions (32 groups of 4 channels each), which captures richer feature representations with fewer parameters.
- Extracts a 2048-dimensional feature vector from each video frame.
- Why ResNeXt? It provides excellent spatial feature extraction — things like skin texture anomalies, blending artifacts, unnatural lighting, and boundary inconsistencies that are telltale signs of deepfakes.

**CBAM (Convolutional Block Attention Module)**
Applied after the ResNeXt backbone, CBAM has two sub-modules:

1. **Channel Attention** — Learns *which* feature channels (out of 2048) are most important for deepfake detection.
   ```
   Input features (B, 2048, H, W)
       |
   Global Average Pooling + Global Max Pooling --> (B, 2048)
       |
   Shared MLP: 2048 -> 128 -> 2048 (reduction ratio = 16)
       |
   Sigmoid activation --> Channel weights
       |
   Multiply with input --> Channel-refined features
   ```

2. **Spatial Attention** — Learns *where* in the image to focus (e.g., face boundaries, eye regions, jawline).
   ```
   Channel-refined features (B, 2048, H, W)
       |
   Channel-wise Average + Max pooling --> (B, 2, H, W)
       |
   7x7 Convolution --> (B, 1, H, W)
       |
   Sigmoid activation --> Spatial attention map
       |
   Multiply with input --> Spatially-refined features
   ```

The attention maps can be visualized to show exactly which facial regions the model focuses on when making its prediction — providing interpretability.

### 2.2 Stage 2: Temporal Modeling (Bidirectional LSTM)

After extracting spatial features from each of the 16 frames independently, we have a sequence:
```
Frame features: (Batch, 16 frames, 2048 features)
```

This sequence is fed into a **2-layer Bidirectional LSTM**:
- Hidden dimension: 512 per direction (1024 total)
- Dropout: 0.3 between layers
- Bidirectional: Processes the frame sequence both forward (frame 1→16) and backward (frame 16→1)
- Output: Concatenated final hidden states from both directions → 1024-dimensional temporal representation
- LayerNorm applied for training stability

**Why temporal modeling?** Deepfakes often have subtle temporal inconsistencies:
- Flickering artifacts between frames
- Unnatural blinking patterns
- Inconsistent head pose transitions
- Temporal boundary artifacts where the face swap occurs

The BiLSTM captures these patterns by modeling how frame-level features evolve over time.

### 2.3 Stage 3: Classification Head

```
Temporal features (1024)
    |
Dropout (0.6) --> Regularization to prevent overfitting
    |
Linear (1024 -> 256) --> Dimensionality reduction
    |
ReLU activation
    |
Dropout (0.3)
    |
Linear (256 -> 2) --> Output logits for [REAL, FAKE]
    |
Softmax --> Probabilities
```

### Complete Model Summary
```
Input:  (Batch, 16 frames, 3 channels, 224 height, 224 width)

Stage 1: ResNeXt-50 + CBAM (per frame)
         (B*16, 3, 224, 224) --> (B*16, 2048)

Stage 2: BiLSTM (across frames)
         (B, 16, 2048) --> (B, 1024)

Stage 3: Classifier
         (B, 1024) --> (B, 2) --> Softmax --> P(REAL), P(FAKE)

Total Parameters: ~40.5 million
```

---

## 3. Data Pipeline

### 3.1 Dataset Preparation

The raw Celeb-DF v2 dataset is split into training/validation/test sets:
```
data/
  train/
    real/   (80% of real videos)
    fake/   (80% of fake videos)
  val/
    real/   (10%)
    fake/   (10%)
  test/
    real/   (10%)
    fake/   (10%)
```

The split is stratified (each class split independently) with a fixed random seed (42) for reproducibility.

### 3.2 Frame Sampling

From each video, exactly **16 frames** are extracted using one of these strategies:

- **Uniform sampling** (validation/inference): Frames are evenly spaced across the video duration.
  - Example: From a 400-frame video, sample at indices [0, 26, 53, 79, 106, ...] to get 16 frames.

- **Uniform with jitter** (training): Same even spacing, but with random offset (±30% of step size) added to each index. This provides data augmentation by showing slightly different frames each epoch.

### 3.3 Face Detection

**Primary detector: MTCNN** (Multi-task Cascaded Convolutional Networks)
- Three-stage cascade detector (P-Net, R-Net, O-Net)
- Detects face bounding boxes and 5 facial landmarks (left eye, right eye, nose, left mouth, right mouth)
- Configured with a 40-pixel margin around the face and minimum face size of 60 pixels
- Selects the highest-confidence face if multiple faces are detected

**Fallback detector: RetinaFace** (from InsightFace)
- Used when MTCNN fails to detect a face
- More robust but slower

**Last resort fallback:** If no face is detected by either method, the frame is center-cropped and resized to 224x224.

### 3.4 Face Alignment

After detection, faces are aligned to a canonical position using landmark-based affine transformation:

1. The 5 detected landmarks are mapped to reference positions:
   ```
   Left eye:    (70, 112)
   Right eye:   (154, 112)
   Nose tip:    (112, 150)
   Mouth left:  (78, 180)
   Mouth right: (146, 180)
   ```
2. An affine transformation matrix is computed using `cv2.estimateAffinePartial2D()`
3. The face is warped to a 224x224 image with consistent orientation

This ensures that all faces have the same position, scale, and orientation regardless of the original video, which makes the model's job easier.

### 3.5 Data Augmentation

**During training**, the following augmentations are applied to each face crop:
- **Random horizontal flip** (50% probability)
- **Random rotation** (±10 degrees)
- **Color jitter** — random changes to brightness (±20%), contrast (±20%), saturation (±10%), hue (±5%)
- **Gaussian blur** (20% probability, kernel size 3)
- **Random erasing** (10% probability) — randomly erases a small rectangular region to simulate occlusion
- **ImageNet normalization** — mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)

**During validation/inference**, only resize and normalization are applied (no randomness).

### 3.6 Pre-extraction Cache

To speed up training, face tensors can be pre-computed and cached:
```
data_cache/
  train/real/video001.pt    (16, 3, 224, 224) tensor
  train/fake/video002.pt
  ...
```

This eliminates the face detection bottleneck during training, providing a 10-50x speedup. The cache is generated by `preextract_faces.py` using 4 parallel workers.

---

## 4. Training Process

### 4.1 Two-Phase Training Strategy

**Phase 1: Backbone Frozen (Epochs 1-15)**
- The ResNeXt-50 backbone weights are frozen (not updated)
- Only CBAM attention, BiLSTM, and classification head are trained
- Learning rate: 1e-4 (0.0001)
- Purpose: Learn the temporal patterns and attention mechanisms without disrupting the pretrained spatial features

**Phase 2: Full Fine-tuning (Epochs 16-40)**
- All parameters are unfrozen and trainable
- The backbone uses a 10x lower learning rate (1e-5) to preserve pretrained features
- CBAM, BiLSTM, and classifier continue training at 1e-4
- Purpose: End-to-end refinement where every layer adapts to the deepfake detection task

This two-phase approach prevents "catastrophic forgetting" — where fine-tuning destroys the useful ImageNet features before the temporal model has learned to use them.

### 4.2 Loss Function

A **combined loss** is used:
```
Total Loss = 0.5 * CrossEntropyLoss + 0.5 * FocalLoss
```

**Cross-Entropy Loss:** Standard classification loss.

**Focal Loss:** Designed for class imbalance scenarios.
```
FL(pt) = -alpha * (1 - pt)^gamma * log(pt)

alpha = 0.25 (weight for the positive class)
gamma = 2.0  (focusing parameter)
```
Focal loss down-weights easy, well-classified examples and focuses training on hard, misclassified examples. This is crucial because some deepfakes are very obvious while others are extremely subtle.

### 4.3 Optimizer & Scheduler

- **Optimizer:** AdamW (Adam with decoupled weight decay)
  - Weight decay: 5e-4 (regularization to prevent overfitting)
  - Parameters are grouped: backbone gets a lower learning rate than the rest

- **Learning Rate Scheduler:** Cosine Annealing
  - Smoothly decays the learning rate following a cosine curve
  - Resets between Phase 1 and Phase 2

- **Gradient Clipping:** Maximum gradient norm of 1.0 to prevent training instability

- **Mixed Precision Training (AMP):** Uses FP16 for forward passes and FP32 for backward passes, reducing GPU memory usage and speeding up training.

### 4.4 Early Stopping

- Monitors validation accuracy after each epoch
- If no improvement for 8 consecutive epochs, training stops
- The best model (by validation accuracy) is saved to `checkpoints/best_model.pt`

### 4.5 Training Configuration Summary

| Parameter          | Value            |
|--------------------|------------------|
| Batch size         | 8                |
| Epochs (max)       | 40               |
| Phase 1 epochs     | 15               |
| Learning rate      | 1e-4             |
| Backbone LR        | 1e-5 (Phase 2)   |
| Optimizer          | AdamW            |
| Weight decay       | 5e-4             |
| Scheduler          | Cosine Annealing |
| Warmup epochs      | 2                |
| Early stopping     | 8 epochs patience|
| Mixed precision    | Enabled          |
| Gradient clipping  | 1.0              |
| Num workers        | 4                |

---

## 5. Evaluation & Results

### 5.1 Metrics Used

- **Accuracy** — Overall percentage of correct predictions
- **F1 Score** — Harmonic mean of precision and recall (handles class imbalance better than accuracy)
- **AUC (Area Under ROC Curve)** — Measures discrimination ability across all thresholds (1.0 = perfect)
- **EER (Equal Error Rate)** — The point where false positive rate equals false negative rate (lower is better)
- **Log Loss** — Measures the quality of predicted probabilities (not just the final decision)

### 5.2 Test Results

Evaluated on **122 test videos** from Celeb-DF v2:

| Metric        | Score  |
|---------------|--------|
| **Accuracy**  | 91.0%  |
| **F1 Score**  | 93.3%  |
| **AUC**       | 97.0%  |
| **EER**       | 9.76%  |
| **Log Loss**  | 0.5675 |

### 5.3 Per-Class Performance

|              | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| **REAL**     | 90%       | 83%    | 86%      |
| **FAKE**     | 92%       | 95%    | 93%      |

### 5.4 Confusion Matrix

```
                    Predicted REAL    Predicted FAKE
Actual REAL              35                7
Actual FAKE               4               76
```

### 5.5 Key Observations
- The model is better at detecting fakes (95% recall) than confirming real videos (83% recall)
- Only 4 fake videos were misclassified as real — very few dangerous misses
- 7 real videos were flagged as fake — some false alarms, but acceptable
- The high AUC of 97% indicates excellent overall discrimination ability
- Best validation accuracy during training: **94.12%**

---

## 6. Inference Pipeline

When a user uploads a video for analysis, here is exactly what happens:

### Step 1: Frame Extraction
16 frames are uniformly sampled from the video using OpenCV.

### Step 2: Face Processing (per frame)
For each of the 16 frames:
1. MTCNN detects the face and returns bounding box + 5 landmarks
2. The face is cropped with a 40px margin
3. Landmarks are used to align the face to a canonical position (224x224)
4. The aligned face is converted from BGR to RGB
5. The face is normalized using ImageNet statistics

### Step 3: Model Inference
```python
tensor = stack(16 processed faces)     # Shape: (16, 3, 224, 224)
batch = tensor.unsqueeze(0)            # Shape: (1, 16, 3, 224, 224)

with torch.no_grad():                  # No gradient computation (faster)
    logits = model(batch)              # Shape: (1, 2)
    probabilities = softmax(logits)    # [P(REAL), P(FAKE)]

fake_probability = probabilities[0, 1] # e.g., 0.87
```

### Step 4: Decision
```
If fake_probability > 0.3:
    prediction = "FAKE"
    confidence = fake_probability       # e.g., 87%
Else:
    prediction = "REAL"
    confidence = 1 - fake_probability   # e.g., 95%
```

The threshold is set to 0.3 (rather than the typical 0.5) to prioritize sensitivity — it's more important to catch deepfakes than to avoid false alarms.

### Step 5: Response
```json
{
    "prediction": "FAKE",
    "confidence": 0.87,
    "fake_probability": 0.87
}
```

---

## 7. Web Application

### 7.1 Backend (FastAPI)

The backend runs on **port 8000** using Uvicorn (ASGI server).

**Endpoints:**
| Endpoint      | Method | Description                          |
|---------------|--------|--------------------------------------|
| `/`           | GET    | API info                             |
| `/health`     | GET    | Health check (model loaded, device)  |
| `/predict`    | POST   | Upload video, get prediction         |
| `/docs`       | GET    | Interactive API documentation        |

**Startup behavior:**
- Loads the trained model from `checkpoints/best_model.pt`
- Initializes the preprocessing pipeline
- Moves model to GPU (CUDA) if available

**Prediction flow:**
1. Receive uploaded video file
2. Validate file type (mp4, avi, mov, mkv, webm)
3. Save to temporary file
4. Run inference pipeline
5. Return JSON result
6. Delete temporary file

### 7.2 Frontend

A single-page HTML application served on **port 3000** with:
- Drag-and-drop video upload
- Visual result display (REAL/FAKE with confidence)
- Fake probability bar chart
- Loading spinner during analysis
- Responsive design with gradient background

---

## 8. Setup & Deployment

### 8.1 Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (recommended, CPU works but slower)
- ~2GB disk space for model and code
- Key dependencies: PyTorch, timm, facenet-pytorch, OpenCV, FastAPI

### 8.2 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare dataset (if training from scratch)
python prepare_dataset.py

# Pre-extract faces for faster training (optional)
python preextract_faces.py

# Train the model
python train.py

# Evaluate on test set
python test.py
```

### 8.3 Running the Application

```bash
# Start backend API
MODEL_CHECKPOINT=checkpoints/best_model.pt python -m uvicorn app.api:app --host 0.0.0.0 --port 8000

# Start frontend (in another terminal)
cd frontend && python -m http.server 3000
```

Then open `http://localhost:3000` in a browser to use the application.

### 8.4 Single Video Inference (Command Line)

```bash
python infer.py --video path/to/video.mp4 --checkpoint checkpoints/best_model.pt --visualize
```

---

## How We Identify Deepfakes — Summary

The system identifies deepfakes by looking for two types of artifacts:

1. **Spatial artifacts** (within individual frames):
   - Skin texture inconsistencies
   - Blending boundaries around the face
   - Unnatural lighting or shadows
   - Eye/teeth rendering anomalies
   - Resolution mismatches between face and background

2. **Temporal artifacts** (across frames):
   - Flickering or jittering of the face
   - Inconsistent blinking patterns
   - Unnatural head movement transitions
   - Frame-to-frame color shifts in the face region

The CBAM attention mechanism helps the model automatically learn which spatial regions are most discriminative, while the BiLSTM captures temporal inconsistencies that are invisible in single frames. Together, they provide a robust deepfake detection system achieving **91% accuracy** and **97% AUC** on the Celeb-DF v2 benchmark.
