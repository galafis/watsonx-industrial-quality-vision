"""Tests for the EfficientNet severity classifier.

Uses mocks for the timm backbone to avoid downloading pretrained weights
during testing. Validates the network architecture, forward pass, and
the high-level DefectClassifier wrapper.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.config import ClassifierConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_mock_backbone(num_features: int = 1408) -> MagicMock:
    """Create a mock timm backbone that returns a fixed-size feature vector."""
    backbone = MagicMock(spec=nn.Module)
    backbone.num_features = num_features

    def forward(x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return torch.randn(batch_size, num_features)

    backbone.__call__ = forward
    backbone.forward = forward
    backbone.parameters = MagicMock(return_value=iter([torch.randn(2, 2, requires_grad=True)]))
    backbone.train = MagicMock(return_value=backbone)
    backbone.eval = MagicMock(return_value=backbone)
    backbone.to = MagicMock(return_value=backbone)
    return backbone


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def classifier_config() -> ClassifierConfig:
    """Default classifier configuration."""
    return ClassifierConfig()


@pytest.fixture
def sample_input() -> torch.Tensor:
    """Batch of 2 images: (B=2, C=3, H=224, W=224)."""
    return torch.randn(2, 3, 224, 224)


@pytest.fixture
def single_input() -> torch.Tensor:
    """Single image: (B=1, C=3, H=224, W=224)."""
    return torch.randn(1, 3, 224, 224)


# ---------------------------------------------------------------------------
# SeverityClassifierNet
# ---------------------------------------------------------------------------


class TestSeverityClassifierNet:
    """Tests for the SeverityClassifierNet architecture."""

    @patch("src.models.classifier.timm")
    def test_forward_output_shape(
        self, mock_timm: MagicMock, classifier_config: ClassifierConfig, sample_input: torch.Tensor
    ) -> None:
        """Forward pass produces (B, num_severity_levels) logits."""
        mock_timm.create_model.return_value = _create_mock_backbone()

        from src.models.classifier import SeverityClassifierNet

        model = SeverityClassifierNet(classifier_config)
        logits = model(sample_input)
        assert logits.shape == (2, 4)

    @patch("src.models.classifier.timm")
    def test_forward_single_image(
        self, mock_timm: MagicMock, classifier_config: ClassifierConfig, single_input: torch.Tensor
    ) -> None:
        """Single image forward pass works correctly."""
        mock_timm.create_model.return_value = _create_mock_backbone()

        from src.models.classifier import SeverityClassifierNet

        model = SeverityClassifierNet(classifier_config)
        logits = model(single_input)
        assert logits.shape == (1, 4)

    @patch("src.models.classifier.timm")
    def test_predict_returns_classes_and_probs(
        self, mock_timm: MagicMock, classifier_config: ClassifierConfig, sample_input: torch.Tensor
    ) -> None:
        """predict() returns class indices and probability distributions."""
        mock_timm.create_model.return_value = _create_mock_backbone()

        from src.models.classifier import SeverityClassifierNet

        model = SeverityClassifierNet(classifier_config)
        classes, probs = model.predict(sample_input)

        assert classes.shape == (2,)
        assert probs.shape == (2, 4)
        # Probabilities should sum to 1 along class dimension
        sums = probs.sum(dim=1)
        torch.testing.assert_close(sums, torch.ones(2), atol=1e-5, rtol=1e-5)

    @patch("src.models.classifier.timm")
    def test_predict_classes_in_valid_range(
        self, mock_timm: MagicMock, classifier_config: ClassifierConfig, sample_input: torch.Tensor
    ) -> None:
        """Predicted class indices are within [0, num_severity_levels)."""
        mock_timm.create_model.return_value = _create_mock_backbone()

        from src.models.classifier import SeverityClassifierNet

        model = SeverityClassifierNet(classifier_config)
        classes, _ = model.predict(sample_input)

        assert (classes >= 0).all()
        assert (classes < classifier_config.num_severity_levels).all()

    @patch("src.models.classifier.timm")
    def test_classifier_head_structure(
        self, mock_timm: MagicMock, classifier_config: ClassifierConfig
    ) -> None:
        """Classification head has Dropout -> Linear -> ReLU -> Dropout -> Linear."""
        mock_timm.create_model.return_value = _create_mock_backbone()

        from src.models.classifier import SeverityClassifierNet

        model = SeverityClassifierNet(classifier_config)
        head = model.classifier

        assert isinstance(head[0], nn.Dropout)
        assert isinstance(head[1], nn.Linear)
        assert head[1].out_features == 256
        assert isinstance(head[2], nn.ReLU)
        assert isinstance(head[3], nn.Dropout)
        assert isinstance(head[4], nn.Linear)
        assert head[4].out_features == classifier_config.num_severity_levels

    @patch("src.models.classifier.timm")
    def test_freeze_backbone(
        self, mock_timm: MagicMock, classifier_config: ClassifierConfig
    ) -> None:
        """freeze_backbone sets requires_grad=False on backbone params."""
        mock_backbone = _create_mock_backbone()
        param = torch.randn(3, 3, requires_grad=True)
        mock_backbone.parameters = MagicMock(return_value=iter([param]))
        mock_timm.create_model.return_value = mock_backbone

        from src.models.classifier import SeverityClassifierNet

        model = SeverityClassifierNet(classifier_config)
        model.freeze_backbone()
        assert not param.requires_grad

    @patch("src.models.classifier.timm")
    def test_unfreeze_backbone(
        self, mock_timm: MagicMock, classifier_config: ClassifierConfig
    ) -> None:
        """unfreeze_backbone restores requires_grad=True."""
        mock_backbone = _create_mock_backbone()
        param = torch.randn(3, 3, requires_grad=False)
        mock_backbone.parameters = MagicMock(return_value=iter([param]))
        mock_timm.create_model.return_value = mock_backbone

        from src.models.classifier import SeverityClassifierNet

        model = SeverityClassifierNet(classifier_config)
        model.unfreeze_backbone()
        assert param.requires_grad

    @patch("src.models.classifier.timm")
    def test_custom_severity_levels(self, mock_timm: MagicMock) -> None:
        """Custom number of severity levels is reflected in output."""
        config = ClassifierConfig(
            num_severity_levels=6,
            severity_labels=("none", "cosmetic", "minor", "moderate", "major", "critical"),
        )
        mock_timm.create_model.return_value = _create_mock_backbone()

        from src.models.classifier import SeverityClassifierNet

        model = SeverityClassifierNet(config)
        x = torch.randn(1, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (1, 6)


# ---------------------------------------------------------------------------
# DefectClassifier wrapper
# ---------------------------------------------------------------------------


class TestDefectClassifier:
    """Tests for the high-level DefectClassifier wrapper."""

    @patch("src.models.classifier.timm")
    def test_classify_returns_severity(self, mock_timm: MagicMock) -> None:
        """classify() returns a dictionary with severity info."""
        mock_timm.create_model.return_value = _create_mock_backbone()

        from src.models.classifier import DefectClassifier

        classifier = DefectClassifier(device="cpu")
        classifier.load()

        image = np.random.rand(224, 224, 3).astype(np.float32)
        result = classifier.classify(image)

        assert "severity" in result
        assert "severity_id" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert result["severity"] in ("cosmetic", "minor", "major", "critical")
        assert 0.0 <= result["confidence"] <= 1.0

    @patch("src.models.classifier.timm")
    def test_classify_probabilities_sum_to_one(self, mock_timm: MagicMock) -> None:
        """Probabilities across severity levels sum to 1."""
        mock_timm.create_model.return_value = _create_mock_backbone()

        from src.models.classifier import DefectClassifier

        classifier = DefectClassifier(device="cpu")
        classifier.load()

        image = np.random.rand(224, 224, 3).astype(np.float32)
        result = classifier.classify(image)

        prob_sum = sum(result["probabilities"].values())
        assert abs(prob_sum - 1.0) < 1e-4

    def test_classify_without_load_raises(self) -> None:
        """classify() before load() raises RuntimeError."""
        from src.models.classifier import DefectClassifier

        classifier = DefectClassifier(device="cpu")
        image = np.random.rand(224, 224, 3).astype(np.float32)

        with pytest.raises(RuntimeError, match="Model not loaded"):
            classifier.classify(image)

    @patch("src.models.classifier.timm")
    def test_device_resolution(self, mock_timm: MagicMock) -> None:
        """Device auto-resolves to cpu or cuda."""
        from src.models.classifier import DefectClassifier

        classifier = DefectClassifier(device="auto")
        assert classifier.device in ("cpu", "cuda")

    @patch("src.models.classifier.timm")
    def test_classify_all_severity_labels_present(self, mock_timm: MagicMock) -> None:
        """All severity labels appear in probabilities dict."""
        mock_timm.create_model.return_value = _create_mock_backbone()

        from src.models.classifier import DefectClassifier

        classifier = DefectClassifier(device="cpu")
        classifier.load()

        image = np.random.rand(224, 224, 3).astype(np.float32)
        result = classifier.classify(image)

        expected_labels = {"cosmetic", "minor", "major", "critical"}
        assert set(result["probabilities"].keys()) == expected_labels
