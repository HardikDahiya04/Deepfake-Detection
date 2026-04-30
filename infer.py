"""Single-video inference script with attention visualization."""

import argparse
import base64
import json
import logging
import math
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model import DeepfakeDetector
from preprocessing.pipeline import PreprocessingPipeline
from evaluation.visualize import plot_attention_heatmap
from utils.config import Config
from utils.helpers import setup_logging, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Deepfake Inference on a single video")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--visualize", action="store_true", help="Save attention heatmaps")
    parser.add_argument("--output-dir", type=str, default="outputs/inference")
    return parser.parse_args()


def _get_video_metadata(video_path: str) -> dict:
    """Extract video metadata using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    return {
        "fps": round(fps, 1),
        "total_frames": total_frames,
        "duration": round(duration, 2),
    }


def _generate_frame_analysis(model, batch, video_path, pipeline) -> dict:
    """Generate per-frame attention scores and heatmap overlays as base64 JPEGs."""
    try:
        with torch.no_grad():
            attn_maps = model.get_attention_maps(batch)  # (1, T, H', W')

        # Per-frame attention intensity as suspicion proxy
        per_frame_scores = attn_maps[0].mean(dim=(-1, -2)).cpu().tolist()

        # Re-sample original frames for overlay
        from preprocessing.frame_sampler import FrameSampler
        sampler = FrameSampler(num_frames=pipeline.num_frames, strategy="uniform")
        frames = sampler.sample(video_path)

        attention_frames = []
        num_frames = min(len(frames), attn_maps.shape[1])
        for i in range(num_frames):
            frame_rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            attn = attn_maps[0, i].detach().cpu().numpy()

            # Resize attention map to frame size
            attn_resized = cv2.resize(attn, (frame_rgb.shape[1], frame_rgb.shape[0]))
            attn_min, attn_max = attn_resized.min(), attn_resized.max()
            attn_norm = (attn_resized - attn_min) / (attn_max - attn_min + 1e-8)

            # Create heatmap overlay
            heatmap = cv2.applyColorMap((attn_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(frame_rgb, 0.6, heatmap_rgb, 0.4, 0)

            # Encode as base64 JPEG
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', overlay_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
            b64 = base64.b64encode(buffer).decode('utf-8')
            attention_frames.append(b64)

        return {
            "per_frame_scores": per_frame_scores[:num_frames],
            "attention_frames": attention_frames,
        }
    except Exception as e:
        return {"error": str(e)}


def predict(video_path: str, model: DeepfakeDetector, pipeline: PreprocessingPipeline,
            device: torch.device, visualize: bool = False, output_dir: str = None,
            include_frames: bool = False) -> dict:
    """Run inference on a single video.

    Returns:
        {"prediction": "REAL" or "FAKE", "confidence": float}
    """
    start_time = time.time()

    # Get video metadata
    meta = _get_video_metadata(video_path)

    # Preprocess
    tensor = pipeline.process_video(video_path)
    if tensor is None:
        return {"prediction": "ERROR", "confidence": 0.0, "error": "Could not process video"}

    # Inference
    batch = tensor.unsqueeze(0).to(device)  # (1, T, C, H, W)
    model.eval()
    with torch.no_grad():
        logits = model(batch)
        # Temperature scaling: softens overconfident predictions (T>1 spreads probabilities)
        # T=2.5 chosen empirically — EER threshold of 0.9967 indicates extreme polarisation
        TEMPERATURE = 2.5
        probs = F.softmax(logits / TEMPERATURE, dim=1)

    fake_prob = probs[0, 1].item()

    # Handle NaN/Inf from untrained or corrupted model weights
    if math.isnan(fake_prob) or math.isinf(fake_prob):
        fake_prob = 0.5  # Uncertain — model has no meaningful weights

    processing_time = time.time() - start_time

    FAKE_THRESHOLD = 0.3  # Lower threshold for better sensitivity to unseen deepfake types
    prediction = "FAKE" if fake_prob > FAKE_THRESHOLD else "REAL"
    confidence = fake_prob if prediction == "FAKE" else 1 - fake_prob

    result = {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "fake_probability": round(fake_prob, 4),
        "processing_details": {
            "frames_analyzed": pipeline.num_frames,
            "total_video_frames": meta["total_frames"],
            "video_duration_seconds": meta["duration"],
            "processing_time_seconds": round(processing_time, 2),
            "device": str(device),
        },
    }

    # Frame-by-frame analysis (optional, for frontend visualization)
    if include_frames:
        result["frame_analysis"] = _generate_frame_analysis(model, batch, video_path, pipeline)

    # Attention visualization to disk (CLI mode)
    if visualize and output_dir:
        try:
            with torch.no_grad():
                attn_maps = model.get_attention_maps(batch)  # (1, T, H', W')
            from preprocessing.frame_sampler import FrameSampler
            sampler = FrameSampler(num_frames=pipeline.num_frames, strategy="uniform")
            frames = sampler.sample(video_path)

            import os
            os.makedirs(output_dir, exist_ok=True)
            for i in range(min(4, len(frames), attn_maps.shape[1])):
                frame_rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
                attn = attn_maps[0, i].cpu().numpy()
                plot_attention_heatmap(
                    frame_rgb, attn,
                    title=f"Frame {i} - {prediction} ({confidence:.2%})",
                    save_path=f"{output_dir}/attention_frame_{i}.png",
                )
        except Exception as e:
            result["visualization_error"] = str(e)

    return result


def main():
    args = parse_args()
    logger = setup_logging()
    device = get_device(args.device)

    # Load model
    cfg = Config()
    cfg.data.num_frames = args.num_frames
    model = DeepfakeDetector(cfg.model)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    logger.info(f"Model loaded from {args.checkpoint}")

    # Build pipeline
    pipeline = PreprocessingPipeline(
        num_frames=args.num_frames,
        is_training=False,
        device=str(device),
    )

    # Predict
    result = predict(args.video, model, pipeline, device,
                     visualize=args.visualize, output_dir=args.output_dir)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
