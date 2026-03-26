"""Image preprocessing utilities for industrial quality inspection.

Handles normalization, resizing, ROI extraction, and color-space conversions
tailored to factory-floor camera setups.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


def resize_with_aspect_ratio(
    image: np.ndarray,
    max_size: int = 1280,
    interpolation: int = cv2.INTER_LINEAR,
) -> tuple[np.ndarray, float]:
    """Resize an image preserving aspect ratio so the longest side
    equals ``max_size``.

    Args:
        image: Input image ``(H, W, C)``.
        max_size: Maximum dimension for the longest side.
        interpolation: OpenCV interpolation method.

    Returns:
        Tuple of (resized_image, scale_factor).
    """
    h, w = image.shape[:2]
    scale = max_size / max(h, w)

    if scale >= 1.0:
        return image.copy(), 1.0

    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    return resized, scale


def normalize_image(
    image: np.ndarray,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Normalize an image with ImageNet mean/std.

    Args:
        image: Input image ``(H, W, 3)`` with values in ``[0, 255]``.
        mean: Per-channel mean.
        std: Per-channel standard deviation.

    Returns:
        Normalized float32 image.
    """
    img = image.astype(np.float32) / 255.0
    img -= np.array(mean, dtype=np.float32)
    img /= np.array(std, dtype=np.float32)
    return img


def extract_roi(
    image: np.ndarray,
    bbox: list[float] | tuple[float, float, float, float],
    padding: float = 0.1,
) -> np.ndarray:
    """Extract a region of interest with optional padding.

    Useful for cropping detected defect regions before classification.

    Args:
        image: Full image ``(H, W, C)``.
        bbox: Bounding box ``[x1, y1, x2, y2]``.
        padding: Fractional padding around the bounding box.

    Returns:
        Cropped ROI image.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = box_w * padding
    pad_y = box_h * padding

    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(w, int(x2 + pad_x))
    y2 = min(h, int(y2 + pad_y))

    return image[y1:y2, x1:x2].copy()


def preprocess_for_detection(
    image: np.ndarray,
    target_size: int = 640,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Preprocess an image for YOLOv8 detection.

    Applies letterbox resizing to maintain aspect ratio with gray padding.

    Args:
        image: Input RGB image ``(H, W, 3)``.
        target_size: Model input dimension.

    Returns:
        Tuple of (preprocessed_image, metadata) where metadata contains
        scale and padding info for coordinate mapping.
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Letterbox with gray padding
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    metadata = {
        "original_size": (h, w),
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "new_size": (new_h, new_w),
    }

    return canvas, metadata


def preprocess_for_classification(
    image: np.ndarray,
    target_size: int = 224,
) -> np.ndarray:
    """Preprocess a cropped defect ROI for the severity classifier.

    Resizes and normalizes with ImageNet statistics.

    Args:
        image: Cropped defect region ``(H, W, 3)`` in RGB.
        target_size: Classifier input dimension.

    Returns:
        Normalized image ``(H, W, 3)`` as float32.
    """
    resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return normalize_image(resized)


def map_boxes_to_original(
    boxes: list[list[float]],
    metadata: dict[str, Any],
) -> list[list[float]]:
    """Map detection bounding boxes back to original image coordinates.

    Args:
        boxes: Detected boxes in letterboxed coordinates ``[x1, y1, x2, y2]``.
        metadata: Preprocessing metadata from ``preprocess_for_detection``.

    Returns:
        Boxes in original image coordinates.
    """
    pad_x = metadata["pad_x"]
    pad_y = metadata["pad_y"]
    scale = metadata["scale"]

    mapped: list[list[float]] = []
    for box in boxes:
        x1 = (box[0] - pad_x) / scale
        y1 = (box[1] - pad_y) / scale
        x2 = (box[2] - pad_x) / scale
        y2 = (box[3] - pad_y) / scale
        mapped.append([x1, y1, x2, y2])
    return mapped
