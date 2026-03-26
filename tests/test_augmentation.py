"""Tests for Albumentations augmentation pipelines.

Validates that training and validation transforms produce correct output
shapes, preserve bounding box consistency, and apply expected augmentations.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import AugmentationConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def aug_config() -> AugmentationConfig:
    """Default augmentation configuration."""
    return AugmentationConfig()


@pytest.fixture
def sample_image() -> np.ndarray:
    """640x640 RGB image for augmentation testing."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_bboxes() -> list[list[float]]:
    """Sample bounding boxes in pascal_voc format [x1, y1, x2, y2]."""
    return [
        [100.0, 100.0, 200.0, 200.0],
        [300.0, 300.0, 450.0, 450.0],
    ]


@pytest.fixture
def sample_labels() -> list[int]:
    """Class labels for the sample bounding boxes."""
    return [0, 2]


# ---------------------------------------------------------------------------
# Training transform
# ---------------------------------------------------------------------------


class TestBuildTrainTransform:
    """Tests for build_train_transform."""

    def test_transform_output_shape(
        self,
        aug_config: AugmentationConfig,
        sample_image: np.ndarray,
        sample_bboxes: list[list[float]],
        sample_labels: list[int],
    ) -> None:
        """Transformed image has correct spatial dimensions."""
        from src.data.augmentation import build_train_transform

        transform = build_train_transform(aug_config, target_size=640)
        result = transform(
            image=sample_image,
            bboxes=sample_bboxes,
            class_labels=sample_labels,
        )
        assert result["image"].shape[:2] == (640, 640)

    def test_transform_preserves_channels(
        self,
        aug_config: AugmentationConfig,
        sample_image: np.ndarray,
        sample_bboxes: list[list[float]],
        sample_labels: list[int],
    ) -> None:
        """Output retains 3 channels."""
        from src.data.augmentation import build_train_transform

        transform = build_train_transform(aug_config, target_size=640)
        result = transform(
            image=sample_image,
            bboxes=sample_bboxes,
            class_labels=sample_labels,
        )
        assert result["image"].shape[2] == 3

    def test_bboxes_present_in_output(
        self,
        aug_config: AugmentationConfig,
        sample_image: np.ndarray,
        sample_bboxes: list[list[float]],
        sample_labels: list[int],
    ) -> None:
        """Bounding boxes are returned (may be filtered by min_visibility)."""
        from src.data.augmentation import build_train_transform

        transform = build_train_transform(aug_config, target_size=640)
        result = transform(
            image=sample_image,
            bboxes=sample_bboxes,
            class_labels=sample_labels,
        )
        assert "bboxes" in result
        assert "class_labels" in result

    def test_output_is_normalized_float(
        self,
        aug_config: AugmentationConfig,
        sample_image: np.ndarray,
        sample_bboxes: list[list[float]],
        sample_labels: list[int],
    ) -> None:
        """Output image is float32 after normalization."""
        from src.data.augmentation import build_train_transform

        transform = build_train_transform(aug_config, target_size=640)
        result = transform(
            image=sample_image,
            bboxes=sample_bboxes,
            class_labels=sample_labels,
        )
        assert result["image"].dtype == np.float32

    def test_custom_target_size(
        self,
        aug_config: AugmentationConfig,
        sample_image: np.ndarray,
        sample_bboxes: list[list[float]],
        sample_labels: list[int],
    ) -> None:
        """Transform respects custom target_size."""
        from src.data.augmentation import build_train_transform

        transform = build_train_transform(aug_config, target_size=320)
        # Need to scale bboxes for smaller image
        scaled_bboxes = [
            [b[0] * 320 / 640, b[1] * 320 / 640, b[2] * 320 / 640, b[3] * 320 / 640]
            for b in sample_bboxes
        ]
        result = transform(
            image=sample_image,
            bboxes=scaled_bboxes,
            class_labels=sample_labels,
        )
        assert result["image"].shape[:2] == (320, 320)

    def test_empty_bboxes(
        self,
        aug_config: AugmentationConfig,
        sample_image: np.ndarray,
    ) -> None:
        """Transform works with no bounding boxes."""
        from src.data.augmentation import build_train_transform

        transform = build_train_transform(aug_config, target_size=640)
        result = transform(image=sample_image, bboxes=[], class_labels=[])
        assert result["image"].shape[:2] == (640, 640)
        assert len(result["bboxes"]) == 0


# ---------------------------------------------------------------------------
# Validation transform
# ---------------------------------------------------------------------------


class TestBuildValTransform:
    """Tests for build_val_transform."""

    def test_val_output_shape(
        self,
        sample_image: np.ndarray,
        sample_bboxes: list[list[float]],
        sample_labels: list[int],
    ) -> None:
        """Validation transform produces correct shape."""
        from src.data.augmentation import build_val_transform

        transform = build_val_transform(target_size=640)
        result = transform(
            image=sample_image,
            bboxes=sample_bboxes,
            class_labels=sample_labels,
        )
        assert result["image"].shape[:2] == (640, 640)

    def test_val_is_deterministic(
        self,
        sample_image: np.ndarray,
        sample_bboxes: list[list[float]],
        sample_labels: list[int],
    ) -> None:
        """Validation transform gives identical results across calls."""
        from src.data.augmentation import build_val_transform

        transform = build_val_transform(target_size=640)
        result1 = transform(
            image=sample_image,
            bboxes=sample_bboxes,
            class_labels=sample_labels,
        )
        result2 = transform(
            image=sample_image,
            bboxes=sample_bboxes,
            class_labels=sample_labels,
        )
        np.testing.assert_array_equal(result1["image"], result2["image"])

    def test_val_normalized(
        self,
        sample_image: np.ndarray,
        sample_bboxes: list[list[float]],
        sample_labels: list[int],
    ) -> None:
        """Validation transform normalizes to float32."""
        from src.data.augmentation import build_val_transform

        transform = build_val_transform(target_size=640)
        result = transform(
            image=sample_image,
            bboxes=sample_bboxes,
            class_labels=sample_labels,
        )
        assert result["image"].dtype == np.float32

    def test_val_bboxes_count_preserved(
        self,
        sample_image: np.ndarray,
        sample_bboxes: list[list[float]],
        sample_labels: list[int],
    ) -> None:
        """All bounding boxes are preserved in validation (no random crop)."""
        from src.data.augmentation import build_val_transform

        transform = build_val_transform(target_size=640)
        result = transform(
            image=sample_image,
            bboxes=sample_bboxes,
            class_labels=sample_labels,
        )
        assert len(result["bboxes"]) == len(sample_bboxes)
        assert len(result["class_labels"]) == len(sample_labels)
