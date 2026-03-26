"""Albumentations augmentation pipelines optimized for industrial imaging.

Industrial defect images have specific characteristics: controlled lighting,
fixed camera angles, metallic/plastic surfaces with reflections, and small
defects that must survive augmentation without being erased.
"""

from __future__ import annotations

import albumentations as A
import structlog
from albumentations.pytorch import ToTensorV2

from src.config import AugmentationConfig

logger = structlog.get_logger(__name__)


def build_train_transform(
    cfg: AugmentationConfig,
    target_size: int = 640,
) -> A.Compose:
    """Build training augmentation pipeline for industrial defect images.

    The pipeline is designed to simulate real-world variations in factory
    imaging conditions: lighting changes, camera vibration, lens blur,
    surface reflections, and slight perspective shifts.

    Args:
        cfg: Augmentation configuration parameters.
        target_size: Output image dimension (square).

    Returns:
        Albumentations ``Compose`` pipeline with bounding-box support.
    """
    transform = A.Compose(
        [
            A.Resize(target_size, target_size),
            # Geometric transforms
            A.HorizontalFlip(p=cfg.horizontal_flip),
            A.VerticalFlip(p=cfg.vertical_flip),
            A.Rotate(limit=cfg.rotation_limit, border_mode=0, p=0.5),
            A.Perspective(scale=(0.02, cfg.perspective_scale), p=0.3),
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(-0.05, 0.05),
                rotate=(-5, 5),
                shear=(-3, 3),
                p=0.3,
            ),
            # Photometric transforms (simulate lighting variation)
            A.RandomBrightnessContrast(
                brightness_limit=cfg.brightness_limit,
                contrast_limit=cfg.contrast_limit,
                p=0.6,
            ),
            A.CLAHE(clip_limit=cfg.clahe_clip_limit, p=0.3),
            A.RandomGamma(gamma_limit=(85, 115), p=0.3),
            # Noise and blur (simulate camera sensor noise / vibration)
            A.GaussNoise(
                var_limit=cfg.gaussian_noise_var_limit,
                p=0.4,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=cfg.gaussian_blur_limit, p=1.0),
                    A.MotionBlur(blur_limit=cfg.motion_blur_limit, p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                ],
                p=0.3,
            ),
            # Surface/lighting effects
            A.RandomShadow(p=cfg.random_shadow),
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=0.2),
            # Normalize to ImageNet stats (for transfer learning)
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )

    logger.info("train_augmentation_built", num_transforms=len(transform.transforms))
    return transform


def build_val_transform(target_size: int = 640) -> A.Compose:
    """Build validation/inference augmentation pipeline.

    Only applies deterministic resize and normalization.

    Args:
        target_size: Output image dimension (square).

    Returns:
        Albumentations ``Compose`` pipeline with bounding-box support.
    """
    return A.Compose(
        [
            A.Resize(target_size, target_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )
