"""PyTorch Dataset for industrial defect images with COCO-format annotations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import structlog
import torch
from torch.utils.data import Dataset

logger = structlog.get_logger(__name__)


class DefectDataset(Dataset):
    """PyTorch Dataset for loading defect images with bounding-box annotations.

    Expects COCO-format annotation JSON with ``images``, ``annotations``,
    and ``categories`` keys.

    Args:
        image_dir: Directory containing defect images.
        annotation_path: Path to COCO-format annotation JSON.
        transform: Optional callable for image augmentation.
        target_size: Resize images to ``(target_size, target_size)``.
    """

    def __init__(
        self,
        image_dir: str | Path,
        annotation_path: str | Path,
        transform: Any | None = None,
        target_size: int = 640,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.target_size = target_size

        with open(annotation_path, encoding="utf-8") as fh:
            coco_data = json.load(fh)

        self.images: list[dict[str, Any]] = coco_data.get("images", [])
        self.annotations: list[dict[str, Any]] = coco_data.get("annotations", [])
        self.categories: list[dict[str, Any]] = coco_data.get("categories", [])

        # Build lookup: image_id -> list of annotations
        self._ann_by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            self._ann_by_image.setdefault(img_id, []).append(ann)

        # Category id -> name mapping
        self.cat_id_to_name: dict[int, str] = {c["id"]: c["name"] for c in self.categories}

        logger.info(
            "dataset_loaded",
            num_images=len(self.images),
            num_annotations=len(self.annotations),
            num_categories=len(self.categories),
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load image and annotations at the given index.

        Returns:
            Dictionary with keys:
                - ``image``: Tensor ``(C, H, W)``
                - ``boxes``: Tensor ``(N, 4)`` in ``[x1, y1, x2, y2]`` format
                - ``labels``: Tensor ``(N,)`` category IDs
                - ``image_id``: int
                - ``file_name``: str
        """
        img_info = self.images[idx]
        image_id: int = img_info["id"]
        file_name: str = img_info["file_name"]

        img_path = self.image_dir / file_name
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("image_load_failed", path=str(img_path))
            image = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = image.shape[:2]

        # Collect annotations for this image
        anns = self._ann_by_image.get(image_id, [])
        boxes: list[list[float]] = []
        labels: list[int] = []

        for ann in anns:
            x, y, w, h = ann["bbox"]  # COCO: [x, y, width, height]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        # Apply augmentation if provided
        if self.transform is not None:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                class_labels=labels,
            )
            image = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["class_labels"]
        else:
            image = cv2.resize(image, (self.target_size, self.target_size))
            # Scale bounding boxes
            scale_x = self.target_size / orig_w
            scale_y = self.target_size / orig_h
            boxes = [
                [b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y] for b in boxes
            ]

        # Convert to tensors
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        boxes_tensor = (
            torch.tensor(boxes, dtype=torch.float32)
            if boxes
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        labels_tensor = (
            torch.tensor(labels, dtype=torch.long)
            if labels
            else torch.zeros((0,), dtype=torch.long)
        )

        return {
            "image": image_tensor,
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": image_id,
            "file_name": file_name,
        }

    @property
    def class_names(self) -> list[str]:
        """Return ordered list of category names."""
        return [self.cat_id_to_name[c["id"]] for c in self.categories]
