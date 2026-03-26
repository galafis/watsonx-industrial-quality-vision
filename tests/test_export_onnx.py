"""Tests for ONNX model export utilities.

Validates the classifier-to-ONNX export pipeline including shape
validation, output file creation, and PyTorch-ONNX output comparison.
Uses mocks for onnxruntime and a lightweight fake classifier to avoid
heavyweight dependencies during unit testing.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.config import ClassifierConfig, InferenceConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeClassifier(nn.Module):
    """Minimal classifier matching SeverityClassifierNet's interface."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 224 * 224, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.view(x.size(0), -1))

    def set_eval_mode(self) -> None:
        self.train(False)

    def cpu(self) -> _FakeClassifier:
        return super().cpu()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def classifier_config() -> ClassifierConfig:
    """Default classifier configuration."""
    return ClassifierConfig()


@pytest.fixture
def inference_config() -> InferenceConfig:
    """Default inference configuration."""
    return InferenceConfig()


@pytest.fixture
def fake_model() -> _FakeClassifier:
    """Lightweight model for export testing."""
    return _FakeClassifier(num_classes=4)


@pytest.fixture
def tmp_onnx_path(tmp_path: Path) -> Path:
    """Temporary path for ONNX file output."""
    return tmp_path / "test_model.onnx"


# ---------------------------------------------------------------------------
# export_classifier_to_onnx
# ---------------------------------------------------------------------------


class TestExportClassifierToOnnx:
    """Tests for export_classifier_to_onnx."""

    @patch("src.models.export_onnx._validate_onnx_model")
    def test_export_creates_file(
        self,
        mock_validate: MagicMock,
        fake_model: _FakeClassifier,
        tmp_onnx_path: Path,
        classifier_config: ClassifierConfig,
        inference_config: InferenceConfig,
    ) -> None:
        """ONNX file is created at the specified path."""
        from src.models.export_onnx import export_classifier_to_onnx

        result_path = export_classifier_to_onnx(
            model=fake_model,
            output_path=tmp_onnx_path,
            config=classifier_config,
            inference_config=inference_config,
        )
        assert result_path.exists()
        assert result_path.suffix == ".onnx"

    @patch("src.models.export_onnx._validate_onnx_model")
    def test_export_returns_path(
        self,
        mock_validate: MagicMock,
        fake_model: _FakeClassifier,
        tmp_onnx_path: Path,
    ) -> None:
        """Function returns the output path."""
        from src.models.export_onnx import export_classifier_to_onnx

        result = export_classifier_to_onnx(
            model=fake_model,
            output_path=tmp_onnx_path,
        )
        assert result == tmp_onnx_path

    @patch("src.models.export_onnx._validate_onnx_model")
    def test_export_calls_validation(
        self,
        mock_validate: MagicMock,
        fake_model: _FakeClassifier,
        tmp_onnx_path: Path,
    ) -> None:
        """Validation function is called after export."""
        from src.models.export_onnx import export_classifier_to_onnx

        export_classifier_to_onnx(
            model=fake_model,
            output_path=tmp_onnx_path,
        )
        mock_validate.assert_called_once()

    @patch("src.models.export_onnx._validate_onnx_model")
    def test_export_creates_parent_dirs(
        self,
        mock_validate: MagicMock,
        fake_model: _FakeClassifier,
        tmp_path: Path,
    ) -> None:
        """Parent directories are created if they don't exist."""
        from src.models.export_onnx import export_classifier_to_onnx

        deep_path = tmp_path / "nested" / "dir" / "model.onnx"
        export_classifier_to_onnx(
            model=fake_model,
            output_path=deep_path,
        )
        assert deep_path.exists()

    def test_export_shape_mismatch_raises(
        self,
        tmp_onnx_path: Path,
    ) -> None:
        """ValueError is raised when output shape is unexpected."""
        from src.models.export_onnx import export_classifier_to_onnx

        bad_model = _FakeClassifier(num_classes=10)
        config = ClassifierConfig(num_severity_levels=4)

        with pytest.raises(ValueError, match="Output shape mismatch"):
            export_classifier_to_onnx(
                model=bad_model,
                output_path=tmp_onnx_path,
                config=config,
            )

    @patch("src.models.export_onnx._validate_onnx_model")
    def test_model_set_to_eval_mode(
        self,
        mock_validate: MagicMock,
        fake_model: _FakeClassifier,
        tmp_onnx_path: Path,
    ) -> None:
        """Model is set to eval mode before export."""
        from src.models.export_onnx import export_classifier_to_onnx

        export_classifier_to_onnx(
            model=fake_model,
            output_path=tmp_onnx_path,
        )
        assert not fake_model.training

    @patch("src.models.export_onnx._validate_onnx_model")
    def test_export_file_nonzero_size(
        self,
        mock_validate: MagicMock,
        fake_model: _FakeClassifier,
        tmp_onnx_path: Path,
    ) -> None:
        """Exported ONNX file has non-zero size."""
        from src.models.export_onnx import export_classifier_to_onnx

        export_classifier_to_onnx(
            model=fake_model,
            output_path=tmp_onnx_path,
        )
        assert tmp_onnx_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Dynamic axes configuration
# ---------------------------------------------------------------------------


class TestOnnxDynamicAxes:
    """Tests for ONNX dynamic axes configuration."""

    @patch("src.models.export_onnx._validate_onnx_model")
    @patch("torch.onnx.export")
    def test_export_with_dynamic_axes(
        self,
        mock_onnx_export: MagicMock,
        mock_validate: MagicMock,
        fake_model: _FakeClassifier,
        tmp_onnx_path: Path,
    ) -> None:
        """Dynamic batch axes are set when onnx_dynamic_axes=True."""
        from src.models.export_onnx import export_classifier_to_onnx

        inf_cfg = InferenceConfig(onnx_dynamic_axes=True)
        export_classifier_to_onnx(
            model=fake_model,
            output_path=tmp_onnx_path,
            inference_config=inf_cfg,
        )

        kwargs = mock_onnx_export.call_args[1]
        assert kwargs["dynamic_axes"] is not None
        assert "input" in kwargs["dynamic_axes"]
        assert "output" in kwargs["dynamic_axes"]

    @patch("src.models.export_onnx._validate_onnx_model")
    @patch("torch.onnx.export")
    def test_export_without_dynamic_axes(
        self,
        mock_onnx_export: MagicMock,
        mock_validate: MagicMock,
        fake_model: _FakeClassifier,
        tmp_onnx_path: Path,
    ) -> None:
        """Dynamic axes are None when onnx_dynamic_axes=False."""
        from src.models.export_onnx import export_classifier_to_onnx

        inf_cfg = InferenceConfig(onnx_dynamic_axes=False)
        export_classifier_to_onnx(
            model=fake_model,
            output_path=tmp_onnx_path,
            inference_config=inf_cfg,
        )

        kwargs = mock_onnx_export.call_args[1]
        assert kwargs["dynamic_axes"] is None

    @patch("src.models.export_onnx._validate_onnx_model")
    @patch("torch.onnx.export")
    def test_export_io_names(
        self,
        mock_onnx_export: MagicMock,
        mock_validate: MagicMock,
        fake_model: _FakeClassifier,
        tmp_onnx_path: Path,
    ) -> None:
        """Export uses 'input' and 'output' as ONNX tensor names."""
        from src.models.export_onnx import export_classifier_to_onnx

        export_classifier_to_onnx(
            model=fake_model,
            output_path=tmp_onnx_path,
        )

        kwargs = mock_onnx_export.call_args[1]
        assert kwargs["input_names"] == ["input"]
        assert kwargs["output_names"] == ["output"]

    @patch("src.models.export_onnx._validate_onnx_model")
    @patch("torch.onnx.export")
    def test_export_opset_version(
        self,
        mock_onnx_export: MagicMock,
        mock_validate: MagicMock,
        fake_model: _FakeClassifier,
        tmp_onnx_path: Path,
        inference_config: InferenceConfig,
    ) -> None:
        """ONNX opset version comes from InferenceConfig."""
        from src.models.export_onnx import export_classifier_to_onnx

        export_classifier_to_onnx(
            model=fake_model,
            output_path=tmp_onnx_path,
            inference_config=inference_config,
        )

        kwargs = mock_onnx_export.call_args[1]
        assert kwargs["opset_version"] == inference_config.onnx_opset_version

    @patch("src.models.export_onnx._validate_onnx_model")
    @patch("torch.onnx.export")
    def test_export_input_shape(
        self,
        mock_onnx_export: MagicMock,
        mock_validate: MagicMock,
        fake_model: _FakeClassifier,
        tmp_onnx_path: Path,
        classifier_config: ClassifierConfig,
    ) -> None:
        """Dummy input has shape (1, 3, input_size, input_size)."""
        from src.models.export_onnx import export_classifier_to_onnx

        export_classifier_to_onnx(
            model=fake_model,
            output_path=tmp_onnx_path,
            config=classifier_config,
        )

        args = mock_onnx_export.call_args[0]
        dummy_input = args[1]
        expected = (1, 3, classifier_config.input_size, classifier_config.input_size)
        assert dummy_input.shape == expected


# ---------------------------------------------------------------------------
# _validate_onnx_model
# ---------------------------------------------------------------------------


class TestValidateOnnxModel:
    """Tests for ONNX model validation."""

    def test_validation_passes_matching_outputs(
        self,
        tmp_onnx_path: Path,
    ) -> None:
        """Validation passes when PyTorch and ONNX outputs match."""
        dummy_input = torch.randn(1, 3, 224, 224)
        pytorch_output = torch.randn(1, 4)

        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_input.shape = [1, 3, 224, 224]
        mock_output = MagicMock()
        mock_output.name = "output"
        mock_output.shape = [1, 4]

        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.run.return_value = [pytorch_output.numpy()]

        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session

        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            from src.models.export_onnx import _validate_onnx_model

            # Should not raise
            _validate_onnx_model(tmp_onnx_path, dummy_input, pytorch_output)

    def test_validation_fails_divergent_outputs(
        self,
        tmp_onnx_path: Path,
    ) -> None:
        """Validation raises ValueError when outputs diverge."""
        dummy_input = torch.randn(1, 3, 224, 224)
        pytorch_output = torch.zeros(1, 4)

        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_input.shape = [1, 3, 224, 224]
        mock_output = MagicMock()
        mock_output.name = "output"
        mock_output.shape = [1, 4]

        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        # Return wildly different output
        mock_session.run.return_value = [np.ones((1, 4), dtype=np.float32) * 100.0]

        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session

        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            from src.models.export_onnx import _validate_onnx_model

            with pytest.raises(ValueError, match="ONNX validation failed"):
                _validate_onnx_model(tmp_onnx_path, dummy_input, pytorch_output, tolerance=1e-4)


# ---------------------------------------------------------------------------
# export_detector_to_onnx
# ---------------------------------------------------------------------------


class TestExportDetectorToOnnx:
    """Tests for YOLOv8 detector ONNX export."""

    @patch("src.models.export_onnx.YOLO", create=True)
    def test_export_detector_calls_ultralytics(
        self,
        mock_yolo_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """YOLO export method is called with correct parameters."""
        mock_model = MagicMock()
        mock_model.export.return_value = str(tmp_path / "detector.onnx")
        mock_yolo_cls.return_value = mock_model

        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
            from src.models.export_onnx import export_detector_to_onnx

            export_detector_to_onnx(
                model_path="weights/best.pt",
                output_path=tmp_path,
                imgsz=640,
                opset=17,
            )

            mock_model.export.assert_called_once_with(format="onnx", imgsz=640, opset=17)


# ---------------------------------------------------------------------------
# Integration-style tests
# ---------------------------------------------------------------------------


class TestOnnxExportIntegration:
    """Integration tests with real ONNX export (no torch.onnx.export mock)."""

    @patch("src.models.export_onnx._validate_onnx_model")
    def test_full_export_pipeline(
        self,
        mock_validate: MagicMock,
        fake_model: _FakeClassifier,
        tmp_path: Path,
    ) -> None:
        """Full pipeline: model -> ONNX file -> validation called."""
        from src.models.export_onnx import export_classifier_to_onnx

        onnx_path = tmp_path / "full_test.onnx"
        result = export_classifier_to_onnx(
            model=fake_model,
            output_path=onnx_path,
            config=ClassifierConfig(),
            inference_config=InferenceConfig(),
        )

        assert result.exists()
        assert result.stat().st_size > 0
        mock_validate.assert_called_once()

    @patch("src.models.export_onnx._validate_onnx_model")
    def test_export_with_dynamic_axes_integration(
        self,
        mock_validate: MagicMock,
        fake_model: _FakeClassifier,
        tmp_path: Path,
    ) -> None:
        """Export with dynamic batch produces valid ONNX file."""
        from src.models.export_onnx import export_classifier_to_onnx

        result = export_classifier_to_onnx(
            model=fake_model,
            output_path=tmp_path / "dynamic.onnx",
            config=ClassifierConfig(),
            inference_config=InferenceConfig(onnx_dynamic_axes=True),
        )
        assert result.exists()

    @patch("src.models.export_onnx._validate_onnx_model")
    def test_export_without_dynamic_axes_integration(
        self,
        mock_validate: MagicMock,
        fake_model: _FakeClassifier,
        tmp_path: Path,
    ) -> None:
        """Export without dynamic axes produces valid ONNX file."""
        from src.models.export_onnx import export_classifier_to_onnx

        result = export_classifier_to_onnx(
            model=fake_model,
            output_path=tmp_path / "static.onnx",
            config=ClassifierConfig(),
            inference_config=InferenceConfig(onnx_dynamic_axes=False),
        )
        assert result.exists()
