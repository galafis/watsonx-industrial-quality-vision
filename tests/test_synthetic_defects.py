"""Tests for synthetic defect generation via cutout/paste augmentation.

Validates the SyntheticDefect dataclass, patch extraction, alpha-blended
pasting, and batch generation pipeline.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.data.synthetic_defects import SyntheticDefect, SyntheticDefectGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_patch() -> np.ndarray:
    """A small 32x32 RGB patch representing a defect."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (32, 32, 3), dtype=np.uint8)


@pytest.fixture
def sample_mask() -> np.ndarray:
    """Binary mask for a 32x32 patch."""
    return np.ones((32, 32), dtype=np.uint8) * 255


@pytest.fixture
def background_image() -> np.ndarray:
    """A 640x640 clean background image."""
    rng = np.random.RandomState(0)
    return rng.randint(100, 200, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_annotated_image() -> np.ndarray:
    """A 480x640 image with distinguishable regions for patch extraction."""
    rng = np.random.RandomState(99)
    return rng.randint(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def generator(tmp_path: Path) -> SyntheticDefectGenerator:
    """SyntheticDefectGenerator with temporary directories."""
    patches_dir = tmp_path / "patches"
    backgrounds_dir = tmp_path / "backgrounds"
    patches_dir.mkdir()
    backgrounds_dir.mkdir()
    return SyntheticDefectGenerator(
        defect_patches_dir=patches_dir,
        clean_backgrounds_dir=backgrounds_dir,
        output_size=640,
    )


# ---------------------------------------------------------------------------
# SyntheticDefect dataclass
# ---------------------------------------------------------------------------

class TestSyntheticDefect:
    """Tests for the SyntheticDefect dataclass."""

    def test_creation(self, sample_patch: np.ndarray, sample_mask: np.ndarray) -> None:
        """SyntheticDefect stores all fields correctly."""
        defect = SyntheticDefect(
            patch=sample_patch,
            mask=sample_mask,
            defect_class="scratch",
            source_file="image_001.png",
        )
        assert defect.defect_class == "scratch"
        assert defect.source_file == "image_001.png"
        assert defect.patch.shape == (32, 32, 3)
        assert defect.mask.shape == (32, 32)

    def test_patch_and_mask_match_dimensions(
        self, sample_patch: np.ndarray, sample_mask: np.ndarray
    ) -> None:
        """Patch spatial dimensions match mask dimensions."""
        defect = SyntheticDefect(
            patch=sample_patch,
            mask=sample_mask,
            defect_class="dent",
            source_file="test.png",
        )
        assert defect.patch.shape[:2] == defect.mask.shape[:2]

    def test_different_defect_classes(
        self, sample_patch: np.ndarray, sample_mask: np.ndarray
    ) -> None:
        """All five defect classes can be used."""
        classes = ["scratch", "dent", "crack", "discoloration", "missing_part"]
        for cls_name in classes:
            defect = SyntheticDefect(
                patch=sample_patch,
                mask=sample_mask,
                defect_class=cls_name,
                source_file="test.png",
            )
            assert defect.defect_class == cls_name


# ---------------------------------------------------------------------------
# SyntheticDefectGenerator initialization
# ---------------------------------------------------------------------------

class TestSyntheticDefectGeneratorInit:
    """Tests for generator initialization."""

    def test_initialization(self, generator: SyntheticDefectGenerator) -> None:
        """Generator initializes with correct parameters."""
        assert generator.output_size == 640
        assert generator.patches_dir.exists()
        assert generator.backgrounds_dir.exists()

    def test_empty_patch_cache(self, generator: SyntheticDefectGenerator) -> None:
        """Generator starts with an empty patch cache."""
        assert generator._patch_cache == {}

    def test_custom_output_size(self, tmp_path: Path) -> None:
        """Custom output_size is stored correctly."""
        gen = SyntheticDefectGenerator(
            defect_patches_dir=tmp_path / "p",
            clean_backgrounds_dir=tmp_path / "b",
            output_size=320,
        )
        assert gen.output_size == 320


# ---------------------------------------------------------------------------
# extract_patches
# ---------------------------------------------------------------------------

class TestExtractPatches:
    """Tests for extracting defect patches from annotated images."""

    def test_extract_single_patch(
        self,
        generator: SyntheticDefectGenerator,
        sample_annotated_image: np.ndarray,
    ) -> None:
        """Extracting a single region returns one SyntheticDefect."""
        patches = generator.extract_patches(
            image=sample_annotated_image,
            boxes=[[50.0, 50.0, 150.0, 150.0]],
            labels=["scratch"],
            source_file="test.png",
        )
        assert len(patches) == 1
        assert patches[0].defect_class == "scratch"
        assert patches[0].patch.shape == (100, 100, 3)

    def test_extract_multiple_patches(
        self,
        generator: SyntheticDefectGenerator,
        sample_annotated_image: np.ndarray,
    ) -> None:
        """Multiple bounding boxes produce multiple patches."""
        patches = generator.extract_patches(
            image=sample_annotated_image,
            boxes=[
                [10.0, 10.0, 60.0, 60.0],
                [200.0, 200.0, 350.0, 350.0],
            ],
            labels=["crack", "dent"],
            source_file="multi.png",
        )
        assert len(patches) == 2
        assert patches[0].defect_class == "crack"
        assert patches[1].defect_class == "dent"

    def test_extract_populates_cache(
        self,
        generator: SyntheticDefectGenerator,
        sample_annotated_image: np.ndarray,
    ) -> None:
        """Extracted patches are added to the internal cache."""
        generator.extract_patches(
            image=sample_annotated_image,
            boxes=[[20.0, 20.0, 100.0, 100.0]],
            labels=["discoloration"],
            source_file="cache_test.png",
        )
        assert "discoloration" in generator._patch_cache
        assert len(generator._patch_cache["discoloration"]) == 1

    def test_skip_tiny_boxes(
        self,
        generator: SyntheticDefectGenerator,
        sample_annotated_image: np.ndarray,
    ) -> None:
        """Bounding boxes smaller than 4px are skipped."""
        patches = generator.extract_patches(
            image=sample_annotated_image,
            boxes=[[100.0, 100.0, 102.0, 102.0]],  # 2x2 px
            labels=["scratch"],
            source_file="tiny.png",
        )
        assert len(patches) == 0

    def test_source_file_recorded(
        self,
        generator: SyntheticDefectGenerator,
        sample_annotated_image: np.ndarray,
    ) -> None:
        """Source filename is stored in the SyntheticDefect."""
        patches = generator.extract_patches(
            image=sample_annotated_image,
            boxes=[[10.0, 10.0, 100.0, 100.0]],
            labels=["missing_part"],
            source_file="origin_42.png",
        )
        assert patches[0].source_file == "origin_42.png"

    def test_mask_is_full_white(
        self,
        generator: SyntheticDefectGenerator,
        sample_annotated_image: np.ndarray,
    ) -> None:
        """Extracted mask is fully white (255)."""
        patches = generator.extract_patches(
            image=sample_annotated_image,
            boxes=[[50.0, 50.0, 150.0, 150.0]],
            labels=["crack"],
            source_file="mask_test.png",
        )
        assert np.all(patches[0].mask == 255)


# ---------------------------------------------------------------------------
# paste_defect
# ---------------------------------------------------------------------------

class TestPasteDefect:
    """Tests for pasting defects onto backgrounds."""

    def test_paste_returns_image_and_bbox(
        self,
        generator: SyntheticDefectGenerator,
        background_image: np.ndarray,
        sample_patch: np.ndarray,
        sample_mask: np.ndarray,
    ) -> None:
        """paste_defect returns an augmented image and a bounding box."""
        defect = SyntheticDefect(
            patch=sample_patch,
            mask=sample_mask,
            defect_class="scratch",
            source_file="test.png",
        )
        result_img, bbox = generator.paste_defect(background_image, defect)
        assert result_img.shape == background_image.shape
        assert len(bbox) == 4

    def test_paste_preserves_dtype(
        self,
        generator: SyntheticDefectGenerator,
        background_image: np.ndarray,
        sample_patch: np.ndarray,
        sample_mask: np.ndarray,
    ) -> None:
        """Output image is uint8."""
        defect = SyntheticDefect(
            patch=sample_patch,
            mask=sample_mask,
            defect_class="dent",
            source_file="test.png",
        )
        result_img, _ = generator.paste_defect(background_image, defect)
        assert result_img.dtype == np.uint8

    def test_paste_does_not_modify_background(
        self,
        generator: SyntheticDefectGenerator,
        background_image: np.ndarray,
        sample_patch: np.ndarray,
        sample_mask: np.ndarray,
    ) -> None:
        """Original background image is not modified in-place."""
        original = background_image.copy()
        defect = SyntheticDefect(
            patch=sample_patch,
            mask=sample_mask,
            defect_class="crack",
            source_file="test.png",
        )
        generator.paste_defect(background_image, defect)
        np.testing.assert_array_equal(background_image, original)

    def test_bbox_within_image_bounds(
        self,
        generator: SyntheticDefectGenerator,
        background_image: np.ndarray,
        sample_patch: np.ndarray,
        sample_mask: np.ndarray,
    ) -> None:
        """Returned bounding box stays within image dimensions."""
        defect = SyntheticDefect(
            patch=sample_patch,
            mask=sample_mask,
            defect_class="scratch",
            source_file="test.png",
        )
        h, w = background_image.shape[:2]
        _, bbox = generator.paste_defect(background_image, defect)

        x1, y1, x2, y2 = bbox
        assert x1 >= 0 and y1 >= 0
        assert x2 <= w and y2 <= h

    def test_paste_at_specific_position(
        self,
        generator: SyntheticDefectGenerator,
        background_image: np.ndarray,
        sample_patch: np.ndarray,
        sample_mask: np.ndarray,
    ) -> None:
        """Defect can be placed at a specific position."""
        defect = SyntheticDefect(
            patch=sample_patch,
            mask=sample_mask,
            defect_class="discoloration",
            source_file="test.png",
        )
        _, bbox = generator.paste_defect(
            background_image,
            defect,
            position=(100, 200),
            scale_range=(1.0, 1.0),
        )
        assert bbox[0] == pytest.approx(100.0, abs=5)
        assert bbox[1] == pytest.approx(200.0, abs=5)


# ---------------------------------------------------------------------------
# generate_batch
# ---------------------------------------------------------------------------

class TestGenerateBatch:
    """Tests for batch synthetic defect generation."""

    def test_batch_with_no_cached_patches(
        self,
        generator: SyntheticDefectGenerator,
        background_image: np.ndarray,
    ) -> None:
        """Returns empty list when no patches are cached for the class."""
        results = generator.generate_batch(
            backgrounds=[background_image],
            defect_class="nonexistent",
        )
        assert results == []

    def test_batch_generates_samples(
        self,
        generator: SyntheticDefectGenerator,
        background_image: np.ndarray,
        sample_annotated_image: np.ndarray,
    ) -> None:
        """Batch generation produces augmented images with annotations."""
        # Populate cache first
        generator.extract_patches(
            image=sample_annotated_image,
            boxes=[[50.0, 50.0, 150.0, 150.0]],
            labels=["scratch"],
            source_file="batch_test.png",
        )

        results = generator.generate_batch(
            backgrounds=[background_image, background_image],
            defect_class="scratch",
            defects_per_image=(1, 2),
        )

        assert len(results) == 2
        for item in results:
            assert "image" in item
            assert "boxes" in item
            assert "labels" in item
            assert item["image"].shape == background_image.shape
            assert len(item["boxes"]) >= 1
            assert all(lbl == "scratch" for lbl in item["labels"])

    def test_batch_result_structure(
        self,
        generator: SyntheticDefectGenerator,
        background_image: np.ndarray,
        sample_annotated_image: np.ndarray,
    ) -> None:
        """Each batch item has consistent boxes and labels lengths."""
        generator.extract_patches(
            image=sample_annotated_image,
            boxes=[[30.0, 30.0, 200.0, 200.0]],
            labels=["dent"],
            source_file="struct_test.png",
        )

        results = generator.generate_batch(
            backgrounds=[background_image],
            defect_class="dent",
            defects_per_image=(2, 3),
        )

        for item in results:
            assert len(item["boxes"]) == len(item["labels"])
