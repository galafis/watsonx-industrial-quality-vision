"""YOLOv8 defect detection wrapper.

Detects five defect classes on industrial surfaces:
scratch, dent, crack, discoloration, missing_part.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import torch

from src.config import DetectorConfig

logger = structlog.get_logger(__name__)


@dataclass
class Detection:
    """A single detected defect."""

    bbox: list[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str


@dataclass
class DetectionResult:
    """Results from defect detection on a single image."""

    detections: list[Detection] = field(default_factory=list)
    image_shape: tuple[int, int] = (0, 0)
    inference_time_ms: float = 0.0

    @property
    def num_detections(self) -> int:
        return len(self.detections)


class DefectDetector:
    """YOLOv8-based defect detector with Ultralytics backend.

    Wraps the Ultralytics YOLO model for industrial defect detection,
    providing a clean interface for inference and batch processing.

    Args:
        config: Detector configuration.
        model_path: Path to trained YOLOv8 weights (``.pt``).
        device: Target device (``cpu``, ``cuda``, ``auto``).
    """

    def __init__(
        self,
        config: DetectorConfig | None = None,
        model_path: str | Path | None = None,
        device: str = "auto",
    ) -> None:
        self.config = config or DetectorConfig()
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self._model: Any = None

        logger.info(
            "detector_initialized",
            architecture=self.config.architecture,
            classes=self.config.classes,
            device=self.device,
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve 'auto' device to CUDA if available, else CPU."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def load(self) -> None:
        """Load the YOLO model from disk.

        Raises:
            FileNotFoundError: If ``model_path`` does not exist.
            ImportError: If ultralytics is not installed.
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for YOLOv8. Install with: pip install ultralytics"
            ) from exc

        if self.model_path and Path(self.model_path).exists():
            self._model = YOLO(str(self.model_path))
            logger.info("detector_loaded", path=str(self.model_path))
        else:
            # Load pretrained model for fine-tuning
            self._model = YOLO(f"{self.config.architecture}.pt")
            logger.info("detector_loaded_pretrained", arch=self.config.architecture)

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Run defect detection on a single image.

        Args:
            image: Input image ``(H, W, 3)`` in RGB format.

        Returns:
            ``DetectionResult`` with detected defects.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import time

        start = time.perf_counter()

        results = self._model(
            image,
            conf=self.config.confidence_threshold,
            iou=self.config.nms_threshold,
            device=self.device,
            verbose=False,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        detections: list[Detection] = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                cls_ids = result.boxes.cls.cpu().numpy().astype(int)

                for box, conf, cls_id in zip(boxes, confs, cls_ids, strict=False):
                    class_name = (
                        self.config.classes[cls_id]
                        if cls_id < len(self.config.classes)
                        else f"class_{cls_id}"
                    )
                    detections.append(
                        Detection(
                            bbox=box.tolist(),
                            confidence=float(conf),
                            class_id=int(cls_id),
                            class_name=class_name,
                        )
                    )

        logger.info(
            "detection_complete",
            num_detections=len(detections),
            inference_ms=round(elapsed_ms, 1),
        )

        return DetectionResult(
            detections=detections,
            image_shape=image.shape[:2],
            inference_time_ms=elapsed_ms,
        )

    def detect_batch(self, images: list[np.ndarray]) -> list[DetectionResult]:
        """Run detection on a batch of images.

        Args:
            images: List of input images ``(H, W, 3)`` in RGB.

        Returns:
            List of ``DetectionResult``, one per image.
        """
        return [self.detect(img) for img in images]

    def train(
        self,
        data_yaml: str | Path,
        epochs: int = 100,
        batch_size: int = 16,
        imgsz: int = 640,
        **kwargs: Any,
    ) -> Any:
        """Fine-tune the detector on a custom defect dataset.

        Args:
            data_yaml: Path to YOLO-format data configuration YAML.
            epochs: Number of training epochs.
            batch_size: Training batch size.
            imgsz: Training image size.
            **kwargs: Additional YOLO training arguments.

        Returns:
            Ultralytics training results.
        """
        if self._model is None:
            self.load()

        logger.info(
            "training_started",
            data=str(data_yaml),
            epochs=epochs,
            batch_size=batch_size,
        )

        results = self._model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=self.device,
            **kwargs,
        )
        return results
