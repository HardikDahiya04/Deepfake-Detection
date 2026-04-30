"""FastAPI backend for deepfake video detection."""

import os
import tempfile
import logging
from pathlib import Path

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from model import DeepfakeDetector
from preprocessing.pipeline import PreprocessingPipeline
from infer import predict
from utils.config import Config
from utils.helpers import get_device

logger = logging.getLogger("deepfake")

app = FastAPI(
    title="Deepfake Detection API",
    description="Upload a video to detect if it's real or deepfake",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and pipeline (loaded once at startup)
_model = None
_pipeline = None
_device = None


@app.on_event("startup")
async def load_model():
    """Load model on server startup."""
    global _model, _pipeline, _device

    checkpoint_path = os.environ.get("MODEL_CHECKPOINT", "checkpoints/best_model.pt")
    device_name = os.environ.get("DEVICE", "cuda")

    _device = get_device(device_name)
    cfg = Config()

    _model = DeepfakeDetector(cfg.model)
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        _model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded model from {checkpoint_path}")
    else:
        logger.warning(f"No checkpoint found at {checkpoint_path}, using random weights")

    _model = _model.to(_device)
    _model.eval()

    _pipeline = PreprocessingPipeline(
        num_frames=cfg.data.num_frames,
        is_training=False,
        device=str(_device),
    )
    logger.info("API ready")


@app.post("/predict")
async def predict_video(video: UploadFile = File(...), include_frames: bool = False):
    """Upload a video and get deepfake prediction."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext = Path(video.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}. Use: {allowed}")

    # Save uploaded video to temp file
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            content = await video.read()
            tmp.write(content)
            tmp_path = tmp.name

        result = predict(tmp_path, _model, _pipeline, _device, include_frames=include_frames)
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "device": str(_device) if _device else "none",
    }


@app.get("/")
async def root():
    return {"message": "Deepfake Detection API", "docs": "/docs"}
