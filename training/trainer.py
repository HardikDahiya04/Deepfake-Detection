"""Training loop with phased training strategy."""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from model import DeepfakeDetector
from training.losses import CombinedLoss, FocalLoss
from utils.config import Config
from utils.helpers import EarlyStopping, save_checkpoint

logger = logging.getLogger("deepfake")


class Trainer:
    """Two-phase trainer for the DeepfakeDetector model."""

    def __init__(self, model: DeepfakeDetector, cfg: Config, device: torch.device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.scaler = GradScaler(enabled=cfg.train.use_amp)

        # Loss function
        if cfg.train.focal_loss:
            self.criterion = CombinedLoss(
                focal_alpha=cfg.train.focal_alpha,
                focal_gamma=cfg.train.focal_gamma,
            )
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Metrics tracking
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    def _build_optimizer(self, phase: int):
        """Build optimizer for the current training phase."""
        if phase == 1:
            # Only train CBAM + BiLSTM + classifier
            params = [
                {"params": self.model.spatial.cbam.parameters(), "lr": self.cfg.train.learning_rate},
                {"params": self.model.temporal.parameters(), "lr": self.cfg.train.learning_rate},
                {"params": self.model.classifier.parameters(), "lr": self.cfg.train.learning_rate},
            ]
        else:
            # Fine-tune everything with lower LR for backbone
            params = [
                {"params": self.model.spatial.features.parameters(), "lr": self.cfg.train.learning_rate * 0.1},
                {"params": self.model.spatial.cbam.parameters(), "lr": self.cfg.train.learning_rate},
                {"params": self.model.temporal.parameters(), "lr": self.cfg.train.learning_rate},
                {"params": self.model.classifier.parameters(), "lr": self.cfg.train.learning_rate},
            ]

        if self.cfg.train.optimizer == "adamw":
            return torch.optim.AdamW(params, weight_decay=self.cfg.train.weight_decay)
        else:
            return torch.optim.SGD(params, momentum=self.cfg.train.momentum, weight_decay=self.cfg.train.weight_decay)

    def _build_scheduler(self, optimizer, total_epochs: int):
        """Build learning rate scheduler."""
        if self.cfg.train.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
        elif self.cfg.train.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return None

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            resume_checkpoint: Optional[str] = None) -> Dict:
        """Run the full two-phase training."""
        save_dir = Path(self.cfg.train.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        best_val_acc = 0.0
        resume_phase = 1
        resume_epoch = 1

        # Load checkpoint for resuming
        if resume_checkpoint and Path(resume_checkpoint).exists():
            ckpt = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            best_val_acc = ckpt.get("best_metric", 0.0)
            resume_phase = ckpt.get("phase", 1)
            resume_epoch = ckpt.get("epoch", 1) + 1
            logger.info(f"Resumed from checkpoint: phase={resume_phase}, epoch={resume_epoch}, best_acc={best_val_acc:.4f}")

        # ── Phase 1: Freeze backbone ──
        if resume_phase == 1:
            logger.info("=" * 60)
            logger.info("Phase 1: Training CBAM + BiLSTM (backbone frozen)")
            logger.info("=" * 60)
        self.model.freeze_backbone()
        optimizer = self._build_optimizer(phase=1)
        scheduler = self._build_scheduler(optimizer, self.cfg.train.phase1_epochs)
        early_stop = EarlyStopping(patience=self.cfg.train.early_stopping_patience)

        phase1_start = resume_epoch if resume_phase == 1 else self.cfg.train.phase1_epochs + 1
        for epoch in range(phase1_start, self.cfg.train.phase1_epochs + 1):
            train_loss, train_acc = self._train_epoch(train_loader, optimizer, epoch, "P1")
            val_loss, val_acc = self._validate(val_loader, epoch, "P1")
            if scheduler:
                scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": best_val_acc,
                    "phase": 1,
                }, str(save_dir / "best_model.pt"))
                logger.info(f"  → Saved best model (val_acc={best_val_acc:.4f})")

            if early_stop(val_acc):
                logger.info(f"  Early stopping at epoch {epoch}")
                break

        # ── Phase 2: Fine-tune all ──
        logger.info("=" * 60)
        logger.info("Phase 2: Fine-tuning entire model")
        logger.info("=" * 60)
        self.model.unfreeze_all()
        remaining = self.cfg.train.epochs - self.cfg.train.phase1_epochs
        optimizer = self._build_optimizer(phase=2)
        scheduler = self._build_scheduler(optimizer, remaining)
        early_stop = EarlyStopping(patience=self.cfg.train.early_stopping_patience)

        phase2_start = resume_epoch if resume_phase == 2 else 1
        for epoch in range(phase2_start, remaining + 1):
            train_loss, train_acc = self._train_epoch(train_loader, optimizer, epoch, "P2")
            val_loss, val_acc = self._validate(val_loader, epoch, "P2")
            if scheduler:
                scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": best_val_acc,
                    "phase": 2,
                }, str(save_dir / "best_model.pt"))
                logger.info(f"  → Saved best model (val_acc={best_val_acc:.4f})")

            if early_stop(val_acc):
                logger.info(f"  Early stopping at epoch {epoch}")
                break

        # Save final model
        save_checkpoint({
            "epoch": self.cfg.train.epochs,
            "model_state_dict": self.model.state_dict(),
            "best_metric": best_val_acc,
        }, str(save_dir / "final_model.pt"))

        return self.history

    def _train_epoch(self, loader: DataLoader, optimizer, epoch: int, phase: str):
        """Run one training epoch."""
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch_idx, (videos, labels) in enumerate(loader):
            videos = videos.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(enabled=self.cfg.train.use_amp):
                logits = self.model(videos)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.max_grad_norm)
            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 20 == 0:
                logger.info(f"  [{phase} Epoch {epoch}] Batch {batch_idx + 1}/{len(loader)} "
                            f"Loss={loss.item():.4f} Acc={correct / total:.4f}")

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        logger.info(f"  [{phase} Epoch {epoch}] Train Loss={avg_loss:.4f} Acc={accuracy:.4f}")
        return avg_loss, accuracy

    @torch.no_grad()
    def _validate(self, loader: DataLoader, epoch: int, phase: str):
        """Run validation."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for videos, labels in loader:
            videos = videos.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(enabled=self.cfg.train.use_amp):
                logits = self.model(videos)
                loss = self.criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        logger.info(f"  [{phase} Epoch {epoch}] Val   Loss={avg_loss:.4f} Acc={accuracy:.4f}")
        return avg_loss, accuracy
