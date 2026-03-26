"""Tests for the YOLOv8 DefectDetector.

Uses mocks for the ultralytics YOLO model to avoid downloading weights
during testing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.config import DetectorConfig
from src.models.detector import DefectDetector, Detection, DetectionResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def detector_config() -> DetectorConfig:
    """Standard detector configuration."""
    return DetectorConfig()


@pytest.fixture
def sample_image() -> np.ndarray:
    """Synthetic 640x640 RGB test image."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (640, 640, 3), dtype=np.uint8)


def _create_mock_yolo_result(
    boxes: list[list[float]],
    confidences: list[float],
    class_ids: list[int],
) -> MagicMock:
    """Create a mock Ultralytics YOLO result object."""
    mock_result = MagicMock()
    mock_boxes = MagicMock()

    mock_boxes.xyxy = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
    mock_boxes.conf = (
        torch.tensor(confidences, dtype=torch.float32) if confidences else torch.zeros(0)
    )
    mock_boxes.cls = torch.tensor(class_ids, dtype=torch.float32) if class_ids else torch.zeros(0)

    mock_result.boxes = mock_boxes
    return mock_result


# ---------------------------------------------------------------------------
# Detection / DetectionResult dataclasses
# ---------------------------------------------------------------------------


class TestDetectionDataclasses:
    """Tests for Detection and DetectionResult."""

    def test_detection_fields(self) -> None:
        """Detection stores all fields correctly."""
        det = Detection(
            bbox=[10.0, 20.0, 100.0, 200.0],
            confidence=0.95,
            class_id=0,
            class_name="scratch",
        )
        assert det.bbox == [10.0, 20.0, 100.0, 200.0]
        assert det.confidence == 0.95
        assert det.class_id == 0
        assert det.class_name == "scratch"

    def test_detection_result_num_detections(self) -> None:
        """num_detections property reflects list length."""
        result = DetectionResult(
            detections=[
                Detection(bbox=[0, 0, 1, 1], confidence=0.9, class_id=0, class_name="scratch"),
                Detection(bbox=[2, 2, 3, 3], confidence=0.8, class_id=1, class_name="dent"),
            ],
            image_shape=(640, 640),
        )
        assert result.num_detections == 2

    def test_empty_detection_result(self) -> None:
        """Empty DetectionResult has zero detections."""
        result = DetectionResult()
        assert result.num_detections == 0
        assert result.image_shape == (0, 0)


# ---------------------------------------------------------------------------
# DefectDetector initialization
# ---------------------------------------------------------------------------


class TestDefectDetectorInit:
    """Tests for DefectDetector initialization."""

    def test_default_config(self) -> None:
        """Default config is used when none provided."""
        detector = DefectDetector()
        assert detector.config.architecture == "yolov8m"
        assert len(detector.config.classes) == 5

    def test_custom_config(self, detector_config: DetectorConfig) -> None:
        """Custom config is stored correctly."""
        detector = DefectDetector(config=detector_config)
        assert detector.config == detector_config

    def test_device_resolution_cpu(self) -> None:
        """Explicit 'cpu' device is respected."""
        detector = DefectDetector(device="cpu")
        assert detector.device == "cpu"

    def test_device_resolution_auto(self) -> None:
        """Auto device resolves to cpu or cuda."""
        detector = DefectDetector(device="auto")
        assert detector.device in ("cpu", "cuda")

    def test_model_not_loaded_initially(self) -> None:
        """Model is None before load() is called."""
        detector = DefectDetector()
        assert detector._model is None


# ---------------------------------------------------------------------------
# DefectDetector.load
# ---------------------------------------------------------------------------


class TestDefectDetectorLoad:
    """Tests for loading the YOLO model."""

    @patch("src.models.detector.YOLO", create=True)
    def test_load_pretrained(self, mock_yolo_cls: MagicMock) -> None:
        """Loading without model_path uses pretrained weights."""
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
            detector = DefectDetector()
            detector.load()
            assert detector._model is not None

    def test_detect_without_load_raises(self, sample_image: np.ndarray) -> None:
        """Calling detect() before load() raises RuntimeError."""
        detector = DefectDetector()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            detector.detect(sample_image)


# ---------------------------------------------------------------------------
# DefectDetector.detect
# ---------------------------------------------------------------------------


class TestDefectDetectorDetect:
    """Tests for defect detection inference."""

    def test_detect_returns_detection_result(self, sample_image: np.ndarray) -> None:
        """detect() returns a DetectionResult."""
        detector = DefectDetector(device="cpu")

        mock_result = _create_mock_yolo_result(
            boxes=[[50.0, 50.0, 150.0, 150.0]],
            confidences=[0.92],
            class_ids=[0],
        )
        mock_model = MagicMock(return_value=[mock_result])
        detector._model = mock_model

        result = detector.detect(sample_image)
        assert isinstance(result, DetectionResult)
        assert result.num_detections == 1
        assert result.detections[0].class_name == "scratch"
        assert result.detections[0].confidence == pytest.approx(0.92, abs=1e-4)

    def test_detect_multiple_detections(self, sample_image: np.ndarray) -> None:
        """Multiple detections are returned correctly."""
        detector = DefectDetector(device="cpu")

        mock_result = _create_mock_yolo_result(
            boxes=[
                [10.0, 10.0, 50.0, 50.0],
                [200.0, 200.0, 300.0, 300.0],
                [400.0, 100.0, 500.0, 200.0],
            ],
            confidences=[0.95, 0.87, 0.73],
            class_ids=[0, 2, 4],
        )
        mock_model = MagicMock(return_value=[mock_result])
        detector._model = mock_model

        result = detector.detect(sample_image)
        assert result.num_detections == 3
        class_names = [d.class_name for d in result.detections]
        assert class_names == ["scratch", "crack", "missing_part"]

    def test_detect_no_detections(self, sample_image: np.ndarray) -> None:
        """Empty detections when no defects found."""
        detector = DefectDetector(device="cpu")

        mock_result = _create_mock_yolo_result(
            boxes=[],
            confidences=[],
            class_ids=[],
        )
        mock_model = MagicMock(return_value=[mock_result])
        detector._model = mock_model

        result = detector.detect(sample_image)
        assert result.num_detections == 0
        assert result.detections == []

    def test_detect_image_shape_recorded(self, sample_image: np.ndarray) -> None:
        """Image shape is recorded in the result."""
        detector = DefectDetector(device="cpu")

        mock_result = _create_mock_yolo_result(boxes=[], confidences=[], class_ids=[])
        detector._model = MagicMock(return_value=[mock_result])

        result = detector.detect(sample_image)
        assert result.image_shape == (640, 640)

    def test_detect_inference_time_positive(self, sample_image: np.ndarray) -> None:
        """Inference time is recorded as a positive value."""
        detector = DefectDetector(device="cpu")

        mock_result = _create_mock_yolo_result(boxes=[], confidences=[], class_ids=[])
        detector._model = MagicMock(return_value=[mock_result])

        result = detector.detect(sample_image)
        assert result.inference_time_ms > 0

    def test_unknown_class_id_fallback(self, sample_image: np.ndarray) -> None:
        """Class ID beyond known classes uses fallback name."""
        detector = DefectDetector(device="cpu")

        mock_result = _create_mock_yolo_result(
            boxes=[[10.0, 10.0, 50.0, 50.0]],
            confidences=[0.8],
            class_ids=[99],
        )
        detector._model = MagicMock(return_value=[mock_result])

        result = detector.detect(sample_image)
        assert result.detections[0].class_name == "class_99"


# ---------------------------------------------------------------------------
# DefectDetector.detect_batch
# ---------------------------------------------------------------------------


class TestDefectDetectorBatch:
    """Tests for batch detection."""

    def test_batch_returns_list(self, sample_image: np.ndarray) -> None:
        """detect_batch returns one result per image."""
        detector = DefectDetector(device="cpu")

        mock_result = _create_mock_yolo_result(boxes=[], confidences=[], class_ids=[])
        detector._model = MagicMock(return_value=[mock_result])

        results = detector.detect_batch([sample_image, sample_image])
        assert len(results) == 2
        assert all(isinstance(r, DetectionResult) for r in results)
