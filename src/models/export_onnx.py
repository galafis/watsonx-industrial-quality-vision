"""Export trained models to ONNX format for edge deployment.

Handles both the severity classifier (PyTorch -> ONNX) and the YOLO
detector (Ultralytics export). Includes input/output shape validation
and inference comparison between PyTorch and ONNX outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import structlog
import torch

from src.config import ClassifierConfig, InferenceConfig
from src.models.classifier import SeverityClassifierNet

logger = structlog.get_logger(__name__)


def export_classifier_to_onnx(
    model: SeverityClassifierNet,
    output_path: str | Path,
    config: ClassifierConfig | None = None,
    inference_config: InferenceConfig | None = None,
) -> Path:
    """Export the severity classifier to ONNX format.

    Performs shape validation and optional output comparison between
    the PyTorch model and the exported ONNX model.

    Args:
        model: Trained severity classifier.
        output_path: Destination path for the ``.onnx`` file.
        config: Classifier configuration for input size.
        inference_config: Inference settings for ONNX opset version.

    Returns:
        Path to the exported ONNX model.

    Raises:
        ValueError: If shape validation fails.
    """
    cfg = config or ClassifierConfig()
    inf_cfg = inference_config or InferenceConfig()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.set_eval_mode()
    model.cpu()

    # Create dummy input with correct shape
    dummy_input = torch.randn(1, 3, cfg.input_size, cfg.input_size)

    # Validate PyTorch output shape
    with torch.no_grad():
        pytorch_output = model(dummy_input)

    expected_shape = (1, cfg.num_severity_levels)
    if pytorch_output.shape != expected_shape:
        raise ValueError(
            f"Output shape mismatch: expected {expected_shape}, "
            f"got {pytorch_output.shape}"
        )

    # Dynamic axes for batch dimension
    dynamic_axes: dict[str, dict[int, str]] | None = None
    if inf_cfg.onnx_dynamic_axes:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=inf_cfg.onnx_opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    logger.info(
        "classifier_exported_onnx",
        path=str(output_path),
        opset_version=inf_cfg.onnx_opset_version,
        input_shape=list(dummy_input.shape),
        output_shape=list(pytorch_output.shape),
    )

    # Validate exported model
    _validate_onnx_model(output_path, dummy_input, pytorch_output)

    return output_path


def export_detector_to_onnx(
    model_path: str | Path,
    output_path: str | Path,
    imgsz: int = 640,
    opset: int = 17,
) -> Path:
    """Export a YOLOv8 detector to ONNX via Ultralytics.

    Args:
        model_path: Path to YOLOv8 ``.pt`` weights.
        output_path: Destination directory for the ONNX file.
        imgsz: Input image size.
        opset: ONNX opset version.

    Returns:
        Path to the exported ONNX model.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is required for YOLO export.") from exc

    model = YOLO(str(model_path))
    export_path = model.export(format="onnx", imgsz=imgsz, opset=opset)

    logger.info(
        "detector_exported_onnx",
        source=str(model_path),
        output=str(export_path),
        imgsz=imgsz,
    )
    return Path(export_path)


def _validate_onnx_model(
    onnx_path: Path,
    dummy_input: torch.Tensor,
    pytorch_output: torch.Tensor,
    tolerance: float = 1e-4,
) -> None:
    """Validate ONNX model by comparing outputs with PyTorch.

    Args:
        onnx_path: Path to exported ONNX model.
        dummy_input: Input tensor used for export.
        pytorch_output: Expected output from PyTorch model.
        tolerance: Maximum allowed absolute difference.

    Raises:
        ValueError: If outputs differ beyond tolerance.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime_not_available", msg="Skipping ONNX validation")
        return

    session = ort.InferenceSession(str(onnx_path))

    # Validate input shape
    input_info = session.get_inputs()[0]
    input_shape = input_info.shape
    logger.info("onnx_input_spec", name=input_info.name, shape=input_shape)

    # Validate output shape
    output_info = session.get_outputs()[0]
    output_shape = output_info.shape
    logger.info("onnx_output_spec", name=output_info.name, shape=output_shape)

    # Run inference
    input_np = dummy_input.numpy()
    onnx_output = session.run(None, {input_info.name: input_np})[0]

    # Compare outputs
    pytorch_np = pytorch_output.numpy()
    max_diff = float(np.max(np.abs(onnx_output - pytorch_np)))

    if max_diff > tolerance:
        raise ValueError(
            f"ONNX validation failed: max difference {max_diff:.6f} "
            f"exceeds tolerance {tolerance:.6f}"
        )

    logger.info(
        "onnx_validation_passed",
        max_difference=f"{max_diff:.6f}",
        tolerance=f"{tolerance:.6f}",
    )
