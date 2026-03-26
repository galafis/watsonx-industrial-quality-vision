"""Synthetic defect generation using cutout/paste augmentation.

Addresses the class imbalance problem common in industrial QC -- rare defect
types (e.g., cracks, missing parts) often have very few real samples.  This
module synthesises additional training examples by pasting defect patches onto
clean surface images.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SyntheticDefect:
    """A synthetic defect patch with metadata."""

    patch: np.ndarray
    mask: np.ndarray
    defect_class: str
    source_file: str


class SyntheticDefectGenerator:
    """Generate synthetic defects by extracting patches from annotated images
    and pasting them onto clean backgrounds with randomised blending.

    This is particularly useful for rare defect classes where collecting
    hundreds of real examples is impractical.

    Args:
        defect_patches_dir: Directory containing cropped defect patches.
        clean_backgrounds_dir: Directory containing defect-free surface images.
        output_size: Output image dimension (square).
    """

    def __init__(
        self,
        defect_patches_dir: str | Path,
        clean_backgrounds_dir: str | Path,
        output_size: int = 640,
    ) -> None:
        self.patches_dir = Path(defect_patches_dir)
        self.backgrounds_dir = Path(clean_backgrounds_dir)
        self.output_size = output_size

        self._patch_cache: dict[str, list[SyntheticDefect]] = {}

    def extract_patches(
        self,
        image: np.ndarray,
        boxes: list[list[float]],
        labels: list[str],
        source_file: str = "unknown",
    ) -> list[SyntheticDefect]:
        """Extract defect patches from an annotated image.

        Args:
            image: Source image ``(H, W, 3)`` in RGB.
            boxes: Bounding boxes ``[x1, y1, x2, y2]``.
            labels: Defect class name for each box.
            source_file: Origin filename for traceability.

        Returns:
            List of extracted ``SyntheticDefect`` instances.
        """
        patches: list[SyntheticDefect] = []

        for box, label in zip(boxes, labels, strict=False):
            x1, y1, x2, y2 = [int(c) for c in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

            if x2 - x1 < 4 or y2 - y1 < 4:
                continue

            patch = image[y1:y2, x1:x2].copy()
            mask = np.ones(patch.shape[:2], dtype=np.uint8) * 255

            defect = SyntheticDefect(
                patch=patch,
                mask=mask,
                defect_class=label,
                source_file=source_file,
            )
            patches.append(defect)
            self._patch_cache.setdefault(label, []).append(defect)

        logger.debug("patches_extracted", count=len(patches), source=source_file)
        return patches

    def paste_defect(
        self,
        background: np.ndarray,
        defect: SyntheticDefect,
        position: tuple[int, int] | None = None,
        scale_range: tuple[float, float] = (0.8, 1.2),
        alpha_range: tuple[float, float] = (0.7, 1.0),
    ) -> tuple[np.ndarray, list[float]]:
        """Paste a defect patch onto a background image with blending.

        Args:
            background: Clean surface image ``(H, W, 3)`` in RGB.
            defect: Synthetic defect patch to paste.
            position: ``(x, y)`` top-left corner; random if ``None``.
            scale_range: Random scale factor range for the patch.
            alpha_range: Blending alpha range.

        Returns:
            Tuple of (augmented_image, bbox) where bbox is
            ``[x1, y1, x2, y2]``.
        """
        bg = background.copy()
        bg_h, bg_w = bg.shape[:2]

        # Random scale
        scale = random.uniform(*scale_range)
        patch_h, patch_w = defect.patch.shape[:2]
        new_w = max(4, int(patch_w * scale))
        new_h = max(4, int(patch_h * scale))

        if new_w >= bg_w or new_h >= bg_h:
            new_w = min(new_w, bg_w - 1)
            new_h = min(new_h, bg_h - 1)

        patch_resized = cv2.resize(defect.patch, (new_w, new_h))
        mask_resized = cv2.resize(defect.mask, (new_w, new_h))

        # Random position
        if position is None:
            x = random.randint(0, max(0, bg_w - new_w))
            y = random.randint(0, max(0, bg_h - new_h))
        else:
            x, y = position

        x = min(x, bg_w - new_w)
        y = min(y, bg_h - new_h)

        # Alpha blending
        alpha = random.uniform(*alpha_range)
        mask_float = (mask_resized / 255.0 * alpha)[..., np.newaxis]

        roi = bg[y : y + new_h, x : x + new_w]
        blended = (patch_resized * mask_float + roi * (1 - mask_float)).astype(np.uint8)
        bg[y : y + new_h, x : x + new_w] = blended

        bbox = [float(x), float(y), float(x + new_w), float(y + new_h)]
        return bg, bbox

    def generate_batch(
        self,
        backgrounds: list[np.ndarray],
        defect_class: str,
        defects_per_image: tuple[int, int] = (1, 3),
    ) -> list[dict[str, Any]]:
        """Generate a batch of synthetic defect images.

        Args:
            backgrounds: List of clean surface images.
            defect_class: Target defect class to paste.
            defects_per_image: Min/max defects to paste per image.

        Returns:
            List of dictionaries with ``image``, ``boxes``, and ``labels``.
        """
        available_patches = self._patch_cache.get(defect_class, [])
        if not available_patches:
            logger.warning("no_patches_available", defect_class=defect_class)
            return []

        results: list[dict[str, Any]] = []

        for bg in backgrounds:
            n_defects = random.randint(*defects_per_image)
            image = bg.copy()
            boxes: list[list[float]] = []
            labels: list[str] = []

            for _ in range(n_defects):
                patch = random.choice(available_patches)
                image, bbox = self.paste_defect(image, patch)
                boxes.append(bbox)
                labels.append(defect_class)

            results.append(
                {
                    "image": image,
                    "boxes": boxes,
                    "labels": labels,
                }
            )

        logger.info(
            "synthetic_batch_generated",
            defect_class=defect_class,
            count=len(results),
        )
        return results
