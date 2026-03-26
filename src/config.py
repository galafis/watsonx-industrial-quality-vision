"""Application configuration loaded from environment variables and YAML settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import yaml
from dotenv import load_dotenv

load_dotenv()

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        path: Path to YAML file.

    Returns:
        Parsed YAML as a dictionary.
    """
    if not path.exists():
        logger.warning("config_file_not_found", path=str(path))
        return {}
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# Watsonx settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WatsonxSettings:
    """IBM Watsonx connection parameters."""

    api_key: str = os.getenv("WATSONX_API_KEY", "")
    project_id: str = os.getenv("WATSONX_PROJECT_ID", "")
    url: str = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    generation_model: str = os.getenv("WATSONX_GENERATION_MODEL", "ibm/granite-13b-chat-v2")


# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectorConfig:
    """YOLOv8 detector configuration."""

    architecture: str = "yolov8m"
    input_size: int = 640
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    classes: tuple[str, ...] = (
        "scratch",
        "dent",
        "crack",
        "discoloration",
        "missing_part",
    )


@dataclass(frozen=True)
class ClassifierConfig:
    """Severity classifier configuration."""

    architecture: str = "efficientnet_b2"
    input_size: int = 224
    num_severity_levels: int = 4
    severity_labels: tuple[str, ...] = ("cosmetic", "minor", "major", "critical")


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters."""

    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    early_stopping_patience: int = 15
    lr_scheduler: str = "cosine"
    warmup_epochs: int = 5
    transfer_learning: bool = True
    pretrained_backbone: bool = True


# ---------------------------------------------------------------------------
# Augmentation settings
# ---------------------------------------------------------------------------


@dataclass
class AugmentationConfig:
    """Augmentation pipeline parameters."""

    horizontal_flip: float = 0.5
    vertical_flip: float = 0.2
    rotation_limit: int = 15
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    gaussian_noise_var_limit: tuple[float, float] = (10.0, 50.0)
    gaussian_blur_limit: tuple[int, int] = (3, 7)
    motion_blur_limit: int = 7
    clahe_clip_limit: float = 4.0
    random_shadow: float = 0.3
    perspective_scale: float = 0.05


# ---------------------------------------------------------------------------
# Inference settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InferenceConfig:
    """Inference pipeline parameters."""

    max_image_size: int = 1280
    device: str = os.getenv("DEVICE", "auto")
    batch_size: int = 1
    onnx_opset_version: int = 17
    onnx_dynamic_axes: bool = True
    onnx_optimize: bool = True


# ---------------------------------------------------------------------------
# Governance settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GovernanceConfig:
    """Governance and monitoring parameters."""

    enabled: bool = os.getenv("GOVERNANCE_ENABLED", "true").lower() == "true"
    log_path: str = os.getenv("GOVERNANCE_LOG_PATH", "./logs/governance")
    map_threshold: float = 0.85
    precision_threshold: float = 0.80
    recall_threshold: float = 0.80
    shift_window_size: int = 100
    degradation_threshold: float = float(os.getenv("SHIFT_DEGRADATION_THRESHOLD", "0.10"))
    alert_on_degradation: bool = True


# ---------------------------------------------------------------------------
# Aggregated configuration
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    """Top-level application configuration.

    Merges environment variables with ``config/settings.yaml``.
    """

    watsonx: WatsonxSettings = field(default_factory=WatsonxSettings)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> Settings:
        """Create settings from YAML configuration file.

        Args:
            path: Optional path override; defaults to ``config/settings.yaml``.

        Returns:
            Populated ``Settings`` instance.
        """
        yaml_cfg = _load_yaml(path or CONFIG_PATH)
        model_cfg = yaml_cfg.get("model", {})
        aug_cfg = yaml_cfg.get("augmentation", {}).get("train", {})
        inf_cfg = yaml_cfg.get("inference", {})
        gov_cfg = yaml_cfg.get("governance", {})

        detector_yaml = model_cfg.get("detector", {})
        classifier_yaml = model_cfg.get("classifier", {})
        training_yaml = model_cfg.get("training", {})

        return cls(
            detector=DetectorConfig(
                architecture=detector_yaml.get("architecture", "yolov8m"),
                input_size=detector_yaml.get("input_size", 640),
                confidence_threshold=detector_yaml.get("confidence_threshold", 0.5),
                nms_threshold=detector_yaml.get("nms_threshold", 0.45),
                classes=tuple(detector_yaml.get("classes", DetectorConfig.classes)),
            ),
            classifier=ClassifierConfig(
                architecture=classifier_yaml.get("architecture", "efficientnet_b2"),
                input_size=classifier_yaml.get("input_size", 224),
                num_severity_levels=classifier_yaml.get("num_severity_levels", 4),
                severity_labels=tuple(
                    classifier_yaml.get("severity_labels", ClassifierConfig.severity_labels)
                ),
            ),
            training=TrainingConfig(
                epochs=training_yaml.get("epochs", 100),
                batch_size=training_yaml.get("batch_size", 16),
                learning_rate=training_yaml.get("learning_rate", 0.001),
                weight_decay=training_yaml.get("weight_decay", 0.0005),
                early_stopping_patience=training_yaml.get("early_stopping_patience", 15),
                lr_scheduler=training_yaml.get("lr_scheduler", "cosine"),
                warmup_epochs=training_yaml.get("warmup_epochs", 5),
                transfer_learning=training_yaml.get("transfer_learning", True),
                pretrained_backbone=training_yaml.get("pretrained_backbone", True),
            ),
            augmentation=AugmentationConfig(
                horizontal_flip=aug_cfg.get("horizontal_flip", 0.5),
                vertical_flip=aug_cfg.get("vertical_flip", 0.2),
                rotation_limit=aug_cfg.get("rotation_limit", 15),
                brightness_limit=aug_cfg.get("brightness_limit", 0.2),
                contrast_limit=aug_cfg.get("contrast_limit", 0.2),
            ),
            inference=InferenceConfig(
                max_image_size=inf_cfg.get("max_image_size", 1280),
                device=inf_cfg.get("device", "auto"),
                batch_size=inf_cfg.get("batch_size", 1),
            ),
            governance=GovernanceConfig(
                enabled=gov_cfg.get("enabled", True),
                map_threshold=gov_cfg.get("accuracy", {}).get("map_threshold", 0.85),
                precision_threshold=gov_cfg.get("accuracy", {}).get("precision_threshold", 0.80),
                recall_threshold=gov_cfg.get("accuracy", {}).get("recall_threshold", 0.80),
                shift_window_size=gov_cfg.get("shift_monitoring", {}).get("window_size", 100),
                degradation_threshold=gov_cfg.get("shift_monitoring", {}).get(
                    "degradation_threshold", 0.10
                ),
            ),
        )


def get_settings() -> Settings:
    """Return the application settings singleton.

    Returns:
        Application ``Settings`` loaded from YAML and environment.
    """
    return Settings.from_yaml()
