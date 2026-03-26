"""Training pipeline with transfer learning, early stopping, and LR scheduling.

Provides a complete training loop for the severity classifier, including:
- Transfer learning with frozen/unfrozen backbone phases
- Cosine annealing or step LR scheduling with warmup
- Early stopping based on validation loss
- Checkpoint saving
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from src.config import TrainingConfig
from src.models.classifier import SeverityClassifierNet

logger = structlog.get_logger(__name__)


@dataclass
class TrainingMetrics:
    """Training metrics for a single epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    learning_rate: float
    epoch_time_s: float


@dataclass
class TrainingHistory:
    """Full training history."""

    metrics: list[TrainingMetrics] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = 0

    def add(self, m: TrainingMetrics) -> None:
        self.metrics.append(m)
        if m.val_loss < self.best_val_loss:
            self.best_val_loss = m.val_loss
            self.best_epoch = m.epoch


class ClassifierTrainer:
    """Training pipeline for the severity classifier.

    Implements a two-phase training approach:
    1. Train only the classification head (backbone frozen)
    2. Fine-tune the full network with lower learning rate

    Args:
        model: Severity classifier network.
        config: Training hyperparameters.
        device: Target device.
        output_dir: Directory for checkpoints and logs.
    """

    def __init__(
        self,
        model: SeverityClassifierNet,
        config: TrainingConfig | None = None,
        device: str = "cpu",
        output_dir: str | Path = "outputs",
    ) -> None:
        self.model = model.to(device)
        self.config = config or TrainingConfig()
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss()
        self.history = TrainingHistory()

    def _build_optimizer(self) -> AdamW:
        """Build AdamW optimizer with weight decay."""
        return AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _build_scheduler(
        self, optimizer: AdamW, total_steps: int
    ) -> SequentialLR:
        """Build LR scheduler with linear warmup + cosine annealing."""
        warmup_steps = self.config.warmup_epochs
        main_steps = max(1, total_steps - warmup_steps)

        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(optimizer, T_max=main_steps, eta_min=1e-6)

        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )

    def _train_epoch(
        self, dataloader: DataLoader, optimizer: AdamW
    ) -> tuple[float, float]:
        """Run a single training epoch.

        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.model.set_train_mode()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            images = batch["image"].to(self.device)
            labels = batch["labels"].to(self.device)

            optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    @torch.no_grad()
    def _validate(self, dataloader: DataLoader) -> tuple[float, float]:
        """Run validation.

        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.model.set_eval_mode()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            images = batch["image"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> TrainingHistory:
        """Execute the full training pipeline.

        Phase 1: Train classification head with frozen backbone.
        Phase 2: Fine-tune the entire network.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.

        Returns:
            ``TrainingHistory`` with per-epoch metrics.
        """
        # Phase 1: Transfer learning (frozen backbone)
        if self.config.transfer_learning:
            logger.info("phase1_transfer_learning_start")
            self.model.freeze_backbone()

        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer, self.config.epochs)

        patience_counter = 0

        for epoch in range(1, self.config.epochs + 1):
            start = time.perf_counter()

            # Unfreeze backbone after warmup
            if (
                self.config.transfer_learning
                and epoch == self.config.warmup_epochs + 1
            ):
                logger.info("phase2_full_finetune_start", epoch=epoch)
                self.model.unfreeze_backbone()
                optimizer = self._build_optimizer()
                scheduler = self._build_scheduler(
                    optimizer, self.config.epochs - epoch + 1
                )

            train_loss, train_acc = self._train_epoch(train_loader, optimizer)
            val_loss, val_acc = self._validate(val_loader)
            scheduler.step()

            elapsed = time.perf_counter() - start
            current_lr = optimizer.param_groups[0]["lr"]

            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_accuracy=train_acc,
                val_accuracy=val_acc,
                learning_rate=current_lr,
                epoch_time_s=elapsed,
            )
            self.history.add(metrics)

            logger.info(
                "epoch_complete",
                epoch=epoch,
                train_loss=round(train_loss, 4),
                val_loss=round(val_loss, 4),
                val_acc=round(val_acc, 4),
                lr=f"{current_lr:.2e}",
            )

            # Checkpoint best model
            if val_loss < self.history.best_val_loss:
                patience_counter = 0
                checkpoint_path = self.output_dir / "best_model.pt"
                torch.save(self.model.state_dict(), checkpoint_path)
                logger.info("checkpoint_saved", path=str(checkpoint_path))
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(
                    "early_stopping",
                    epoch=epoch,
                    best_epoch=self.history.best_epoch,
                    best_val_loss=round(self.history.best_val_loss, 4),
                )
                break

        return self.history
