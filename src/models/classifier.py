"""EfficientNet/ResNet classification head for defect type and severity.

Provides a flexible CNN classifier built with ``timm`` that can use
EfficientNet-B2 or ResNet-50 backbones with transfer learning from
ImageNet weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import structlog
import timm
import torch
import torch.nn as nn

from src.config import ClassifierConfig

logger = structlog.get_logger(__name__)


class SeverityClassifierNet(nn.Module):
    """CNN classifier for defect severity using a ``timm`` backbone.

    Architecture:
        Backbone (EfficientNet-B2/ResNet-50)
        -> Global Average Pooling
        -> Dropout(0.3)
        -> FC(num_features, 256)
        -> ReLU + Dropout(0.2)
        -> FC(256, num_classes)

    Args:
        config: Classifier configuration.
    """

    def __init__(self, config: ClassifierConfig | None = None) -> None:
        super().__init__()
        self.config = config or ClassifierConfig()

        # Create backbone from timm
        self.backbone = timm.create_model(
            self.config.architecture,
            pretrained=True,
            num_classes=0,  # Remove default classifier head
        )

        # Get feature dimension from backbone
        num_features = self.backbone.num_features

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, self.config.num_severity_levels),
        )

        logger.info(
            "classifier_created",
            backbone=self.config.architecture,
            num_features=num_features,
            num_classes=self.config.num_severity_levels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor ``(B, 3, H, W)``.

        Returns:
            Logits tensor ``(B, num_severity_levels)``.
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run inference and return class predictions with probabilities.

        Args:
            x: Input tensor ``(B, 3, H, W)``.

        Returns:
            Tuple of (predicted_classes, probabilities).
        """
        self.set_eval_mode()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            classes = torch.argmax(probs, dim=1)
        return classes, probs

    def set_eval_mode(self) -> None:
        """Set the model to evaluation mode."""
        self.train(False)

    def set_train_mode(self) -> None:
        """Set the model to training mode."""
        self.train(True)

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("backbone_frozen", architecture=self.config.architecture)

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters for end-to-end fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("backbone_unfrozen", architecture=self.config.architecture)


class DefectClassifier:
    """High-level wrapper for the severity classification model.

    Handles model loading, inference, and label mapping.

    Args:
        config: Classifier configuration.
        model_path: Path to saved model weights.
        device: Target device (``cpu``, ``cuda``, ``auto``).
    """

    def __init__(
        self,
        config: ClassifierConfig | None = None,
        model_path: str | Path | None = None,
        device: str = "auto",
    ) -> None:
        self.config = config or ClassifierConfig()
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self.model: SeverityClassifierNet | None = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def load(self) -> None:
        """Load or create the classifier model."""
        self.model = SeverityClassifierNet(self.config)

        if self.model_path and Path(self.model_path).exists():
            state_dict = torch.load(
                str(self.model_path),
                map_location=self.device,
                weights_only=True,
            )
            self.model.load_state_dict(state_dict)
            logger.info("classifier_loaded", path=str(self.model_path))

        self.model = self.model.to(self.device)
        self.model.set_eval_mode()

    def classify(self, image: np.ndarray) -> dict[str, Any]:
        """Classify defect severity for a single cropped defect image.

        Args:
            image: Preprocessed image ``(H, W, 3)`` normalized float32.

        Returns:
            Dictionary with ``severity``, ``confidence``, and ``probabilities``.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Convert to tensor: (H, W, C) -> (1, C, H, W)
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        classes, probs = self.model.predict(tensor)
        class_id = int(classes[0].item())
        confidence = float(probs[0, class_id].item())

        severity_label = (
            self.config.severity_labels[class_id]
            if class_id < len(self.config.severity_labels)
            else f"severity_{class_id}"
        )

        return {
            "severity": severity_label,
            "severity_id": class_id,
            "confidence": confidence,
            "probabilities": {
                label: float(probs[0, i].item())
                for i, label in enumerate(self.config.severity_labels)
            },
        }
