"""
Tests for evaluation metrics.
"""

import numpy as np
import pytest
from src.utils.evaluation_metrics import (
    top_k_accuracy,
    compute_far_frr,
    compute_eer,
    compute_ece,
    compute_brier_score,
    selective_risk,
    error_taxonomy,
    per_device_frr,
    compute_confusion_matrix,
)


@pytest.fixture
def perfect_predictions():
    """Perfectly calibrated predictions."""
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_probs = np.zeros((10, 3))
    for i, t in enumerate(y_true):
        y_probs[i, t] = 0.95
        for j in range(3):
            if j != t:
                y_probs[i, j] = 0.025
    return y_true, y_probs


@pytest.fixture
def random_predictions():
    """Random predictions for 5 classes."""
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 5, 100)
    logits = rng.randn(100, 5)
    # Softmax
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    y_probs = exp / exp.sum(axis=1, keepdims=True)
    return y_true, y_probs


class TestTopKAccuracy:

    def test_top1_perfect(self, perfect_predictions):
        y_true, y_probs = perfect_predictions
        acc = top_k_accuracy(y_true, y_probs, k=1)
        assert acc == 1.0

    def test_top5_includes_top1(self, random_predictions):
        y_true, y_probs = random_predictions
        top1 = top_k_accuracy(y_true, y_probs, k=1)
        top5 = top_k_accuracy(y_true, y_probs, k=5)
        assert top5 >= top1

    def test_top_k_equals_n_classes(self, random_predictions):
        y_true, y_probs = random_predictions
        top_all = top_k_accuracy(y_true, y_probs, k=5)
        assert top_all == 1.0  # All classes covered


class TestFARFRR:

    def test_perfect_separation(self):
        genuine = np.array([0.8, 0.9, 0.85, 0.95])
        impostor = np.array([0.1, 0.2, 0.15, 0.05])
        far, frr = compute_far_frr(genuine, impostor, threshold=0.5)
        assert far == 0.0  # No impostor above threshold
        assert frr == 0.0  # No genuine below threshold

    def test_all_accept(self):
        genuine = np.array([0.8, 0.9])
        impostor = np.array([0.7, 0.6])
        far, frr = compute_far_frr(genuine, impostor, threshold=0.0)
        assert far == 1.0  # All impostors accepted
        assert frr == 0.0  # No genuine rejected

    def test_eer_exists(self):
        rng = np.random.RandomState(42)
        genuine = rng.normal(0.7, 0.15, 1000)
        impostor = rng.normal(0.3, 0.15, 1000)
        eer, threshold = compute_eer(genuine, impostor)
        assert 0.0 < eer < 0.5
        assert isinstance(threshold, float)


class TestCalibration:

    def test_ece_perfect(self, perfect_predictions):
        y_true, y_probs = perfect_predictions
        ece = compute_ece(y_true, y_probs)
        # Perfect calibration ≈ 0
        assert ece < 0.1

    def test_ece_range(self, random_predictions):
        y_true, y_probs = random_predictions
        ece = compute_ece(y_true, y_probs)
        assert 0.0 <= ece <= 1.0

    def test_brier_perfect(self, perfect_predictions):
        y_true, y_probs = perfect_predictions
        brier = compute_brier_score(y_true, y_probs)
        # Near-perfect predictions → low Brier score
        assert brier < 0.1

    def test_brier_range(self, random_predictions):
        y_true, y_probs = random_predictions
        brier = compute_brier_score(y_true, y_probs)
        assert brier >= 0.0


class TestSelectiveRisk:

    def test_full_coverage(self, random_predictions):
        y_true, y_probs = random_predictions
        acc, thresh = selective_risk(y_true, y_probs, coverage=1.0)
        # Full coverage = standard accuracy
        preds = np.argmax(y_probs, axis=1)
        expected = float(np.mean(preds == y_true))
        np.testing.assert_almost_equal(acc, expected)

    def test_higher_coverage_lower_accuracy(self, random_predictions):
        """Lower coverage (only confident predictions) should have higher accuracy."""
        y_true, y_probs = random_predictions
        acc_100, _ = selective_risk(y_true, y_probs, coverage=1.0)
        acc_50, _ = selective_risk(y_true, y_probs, coverage=0.50)
        # With 50% coverage (keeping only most confident half), accuracy should
        # generally improve. Allow small tolerance for edge cases.
        assert acc_50 >= acc_100 - 0.05


class TestErrorTaxonomy:

    def test_all_correct_no_errors(self):
        y_true = [0, 1, 2]
        y_pred = [0, 1, 2]
        device_to_model = {0: "A", 1: "B", 2: "C"}
        result = error_taxonomy(y_true, y_pred, device_to_model)
        assert result["total_errors"] == 0

    def test_same_model_errors(self):
        y_true = [0, 1]
        y_pred = [1, 0]
        device_to_model = {0: "Canon_EOS_5D", 1: "Canon_EOS_5D"}
        result = error_taxonomy(y_true, y_pred, device_to_model)
        assert result["same_model_errors"] == 2
        assert result["cross_model_errors"] == 0

    def test_cross_model_errors(self):
        y_true = [0, 1]
        y_pred = [1, 0]
        device_to_model = {0: "Canon", 1: "Nikon"}
        result = error_taxonomy(y_true, y_pred, device_to_model)
        assert result["cross_model_errors"] == 2
        assert result["same_model_errors"] == 0


class TestPerDeviceFRR:

    def test_perfect_classification(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 0, 1, 1, 2, 2]
        frr = per_device_frr(y_true, y_pred)
        assert all(v == 0.0 for v in frr.values())

    def test_worst_case(self):
        y_true = [0, 0, 0]
        y_pred = [1, 1, 1]
        frr = per_device_frr(y_true, y_pred)
        assert frr[0] == 1.0

    def test_partial_errors(self):
        y_true = [0, 0, 0, 0]
        y_pred = [0, 0, 1, 1]
        frr = per_device_frr(y_true, y_pred)
        assert frr[0] == 0.5
