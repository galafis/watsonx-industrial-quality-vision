"""Tests for image preprocessing utilities.

Covers normalization, ROI extraction, color-space conversions,
letterbox preprocessing for detection, and coordinate mapping.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.preprocessing import (
    extract_roi,
    map_boxes_to_original,
    normalize_image,
    preprocess_for_classification,
    preprocess_for_detection,
    resize_with_aspect_ratio,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_image_rgb() -> np.ndarray:
    """Create a synthetic 480x640 RGB image with known pixel values."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def small_image_rgb() -> np.ndarray:
    """Create a small 100x100 image for quick tests."""
    return np.full((100, 100, 3), 128, dtype=np.uint8)


@pytest.fixture
def square_image_rgb() -> np.ndarray:
    """Create a 640x640 square image."""
    rng = np.random.RandomState(0)
    return rng.randint(0, 256, (640, 640, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# resize_with_aspect_ratio
# ---------------------------------------------------------------------------


class TestResizeWithAspectRatio:
    """Tests for resize_with_aspect_ratio."""

    def test_downscale_landscape(self, sample_image_rgb: np.ndarray) -> None:
        """Landscape image is downscaled so longest side equals max_size."""
        resized, scale = resize_with_aspect_ratio(sample_image_rgb, max_size=320)
        h, w = resized.shape[:2]
        assert max(h, w) == 320
        assert scale < 1.0
        assert resized.dtype == sample_image_rgb.dtype

    def test_no_upscale_when_smaller(self, small_image_rgb: np.ndarray) -> None:
        """Images smaller than max_size are returned unchanged."""
        resized, scale = resize_with_aspect_ratio(small_image_rgb, max_size=640)
        assert scale == 1.0
        assert resized.shape == small_image_rgb.shape

    def test_aspect_ratio_preserved(self, sample_image_rgb: np.ndarray) -> None:
        """Aspect ratio is maintained after resize."""
        orig_h, orig_w = sample_image_rgb.shape[:2]
        orig_ratio = orig_w / orig_h
        resized, _ = resize_with_aspect_ratio(sample_image_rgb, max_size=320)
        new_h, new_w = resized.shape[:2]
        new_ratio = new_w / new_h
        assert abs(orig_ratio - new_ratio) < 0.02

    def test_square_image(self, square_image_rgb: np.ndarray) -> None:
        """Square images are resized correctly."""
        resized, _scale = resize_with_aspect_ratio(square_image_rgb, max_size=320)
        h, w = resized.shape[:2]
        assert h == 320
        assert w == 320


# ---------------------------------------------------------------------------
# normalize_image
# ---------------------------------------------------------------------------


class TestNormalizeImage:
    """Tests for normalize_image."""

    def test_output_dtype(self, small_image_rgb: np.ndarray) -> None:
        """Output must be float32."""
        result = normalize_image(small_image_rgb)
        assert result.dtype == np.float32

    def test_output_shape_preserved(self, sample_image_rgb: np.ndarray) -> None:
        """Spatial dimensions are unchanged."""
        result = normalize_image(sample_image_rgb)
        assert result.shape == sample_image_rgb.shape

    def test_imagenet_normalization_range(self, small_image_rgb: np.ndarray) -> None:
        """Normalized values should be roughly within [-3, 3] for ImageNet stats."""
        result = normalize_image(small_image_rgb)
        assert result.min() > -5.0
        assert result.max() < 5.0

    def test_custom_mean_std(self) -> None:
        """Custom mean/std values are applied correctly."""
        image = np.full((10, 10, 3), 127, dtype=np.uint8)
        result = normalize_image(image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        expected = (127.0 / 255.0 - 0.5) / 0.5
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_zero_image(self) -> None:
        """All-black image normalizes to -mean/std."""
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        result = normalize_image(image)
        expected_ch0 = -0.485 / 0.229
        np.testing.assert_allclose(result[:, :, 0], expected_ch0, atol=1e-5)


# ---------------------------------------------------------------------------
# extract_roi
# ---------------------------------------------------------------------------


class TestExtractROI:
    """Tests for extract_roi."""

    def test_basic_extraction(self, sample_image_rgb: np.ndarray) -> None:
        """Basic ROI extraction produces valid sub-image."""
        roi = extract_roi(sample_image_rgb, [100, 50, 300, 200], padding=0.0)
        assert roi.shape == (150, 200, 3)

    def test_padding_expands_region(self, sample_image_rgb: np.ndarray) -> None:
        """Padding expands the extracted region."""
        roi_no_pad = extract_roi(sample_image_rgb, [100, 50, 300, 200], padding=0.0)
        roi_with_pad = extract_roi(sample_image_rgb, [100, 50, 300, 200], padding=0.2)
        assert roi_with_pad.shape[0] >= roi_no_pad.shape[0]
        assert roi_with_pad.shape[1] >= roi_no_pad.shape[1]

    def test_bbox_at_edge(self, sample_image_rgb: np.ndarray) -> None:
        """Bounding box near image edge is clamped correctly."""
        h, w = sample_image_rgb.shape[:2]
        roi = extract_roi(sample_image_rgb, [w - 50, h - 50, w + 50, h + 50], padding=0.1)
        assert roi.shape[0] > 0
        assert roi.shape[1] > 0

    def test_roi_is_copy(self, sample_image_rgb: np.ndarray) -> None:
        """Returned ROI is a copy, not a view."""
        roi = extract_roi(sample_image_rgb, [10, 10, 100, 100], padding=0.0)
        roi[:] = 0
        assert sample_image_rgb[10:100, 10:100].sum() > 0


# ---------------------------------------------------------------------------
# preprocess_for_detection
# ---------------------------------------------------------------------------


class TestPreprocessForDetection:
    """Tests for preprocess_for_detection (letterbox resize)."""

    def test_output_shape(self, sample_image_rgb: np.ndarray) -> None:
        """Output has the target square dimensions."""
        result, _meta = preprocess_for_detection(sample_image_rgb, target_size=640)
        assert result.shape == (640, 640, 3)

    def test_metadata_keys(self, sample_image_rgb: np.ndarray) -> None:
        """Metadata contains expected keys."""
        _, meta = preprocess_for_detection(sample_image_rgb, target_size=640)
        expected_keys = {"original_size", "scale", "pad_x", "pad_y", "new_size"}
        assert expected_keys == set(meta.keys())

    def test_gray_padding_value(self, sample_image_rgb: np.ndarray) -> None:
        """Padded regions use gray value 114."""
        result, meta = preprocess_for_detection(sample_image_rgb, target_size=640)
        pad_y = meta["pad_y"]
        if pad_y > 0:
            top_strip = result[:pad_y, :, :]
            assert np.all(top_strip == 114)

    def test_square_image_no_padding(self, square_image_rgb: np.ndarray) -> None:
        """Square image should have zero padding."""
        _result, meta = preprocess_for_detection(square_image_rgb, target_size=640)
        assert meta["pad_x"] == 0
        assert meta["pad_y"] == 0

    def test_dtype_preserved(self, sample_image_rgb: np.ndarray) -> None:
        """Output dtype is uint8."""
        result, _ = preprocess_for_detection(sample_image_rgb, target_size=640)
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# preprocess_for_classification
# ---------------------------------------------------------------------------


class TestPreprocessForClassification:
    """Tests for preprocess_for_classification."""

    def test_output_shape(self, small_image_rgb: np.ndarray) -> None:
        """Output shape is (224, 224, 3)."""
        result = preprocess_for_classification(small_image_rgb, target_size=224)
        assert result.shape == (224, 224, 3)

    def test_output_is_normalized(self, small_image_rgb: np.ndarray) -> None:
        """Output is normalized float32."""
        result = preprocess_for_classification(small_image_rgb, target_size=224)
        assert result.dtype == np.float32

    def test_custom_target_size(self, small_image_rgb: np.ndarray) -> None:
        """Custom target size is respected."""
        result = preprocess_for_classification(small_image_rgb, target_size=128)
        assert result.shape == (128, 128, 3)


# ---------------------------------------------------------------------------
# map_boxes_to_original
# ---------------------------------------------------------------------------


class TestMapBoxesToOriginal:
    """Tests for map_boxes_to_original."""

    def test_identity_mapping(self) -> None:
        """No padding and scale=1 returns same boxes."""
        boxes = [[100.0, 50.0, 200.0, 150.0]]
        meta = {"pad_x": 0, "pad_y": 0, "scale": 1.0}
        mapped = map_boxes_to_original(boxes, meta)
        np.testing.assert_allclose(mapped[0], boxes[0], atol=1e-5)

    def test_scale_mapping(self) -> None:
        """Scale is correctly applied to coordinates."""
        boxes = [[100.0, 100.0, 200.0, 200.0]]
        meta = {"pad_x": 0, "pad_y": 0, "scale": 0.5}
        mapped = map_boxes_to_original(boxes, meta)
        assert mapped[0] == [200.0, 200.0, 400.0, 400.0]

    def test_padding_offset(self) -> None:
        """Padding offsets are subtracted before scaling."""
        boxes = [[110.0, 60.0, 210.0, 160.0]]
        meta = {"pad_x": 10, "pad_y": 10, "scale": 1.0}
        mapped = map_boxes_to_original(boxes, meta)
        assert mapped[0] == [100.0, 50.0, 200.0, 150.0]

    def test_multiple_boxes(self) -> None:
        """Multiple boxes are all mapped."""
        boxes = [[10.0, 10.0, 20.0, 20.0], [30.0, 30.0, 40.0, 40.0]]
        meta = {"pad_x": 0, "pad_y": 0, "scale": 1.0}
        mapped = map_boxes_to_original(boxes, meta)
        assert len(mapped) == 2

    def test_empty_boxes(self) -> None:
        """Empty box list returns empty result."""
        meta = {"pad_x": 0, "pad_y": 0, "scale": 1.0}
        mapped = map_boxes_to_original([], meta)
        assert mapped == []
