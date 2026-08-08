import pytest
import numpy as np
from accuracy import get_accuracy, Accuracy
from safety import get_safety, Safety
from slicing import get_slicing, Slicing
from report import report, Report

# Test with: cd src/infra/metrics && pytest test_metrics.py -v


# ── Fixtures ──

@pytest.fixture
def perfect_predictions():
    """Predictions that exactly match ground truth."""
    y_true = np.array([[10.0, 5.0], [20.0, -10.0], [30.0, 15.0], [-5.0, 3.0]])
    y_pred = np.array([[10.0, 5.0], [20.0, -10.0], [30.0, 15.0], [-5.0, 3.0]])
    return y_pred, y_true


@pytest.fixture
def known_error_predictions():
    """Predictions with known constant error of 2 degrees on both axes."""
    y_true = np.array([[10.0, 5.0], [20.0, -10.0], [30.0, 15.0], [-5.0, 3.0]])
    y_pred = np.array([[12.0, 7.0], [22.0, -8.0], [32.0, 17.0], [-3.0, 5.0]])
    return y_pred, y_true


@pytest.fixture
def spread_yaw_data():
    """Data with yaw values spread across all bins."""
    y_true = np.array([
        [5.0, 1.0],     # bin 0-15
        [10.0, 2.0],    # bin 0-15
        [20.0, 3.0],    # bin 15-30
        [25.0, 4.0],    # bin 15-30
        [35.0, 5.0],    # bin 30-45
        [40.0, 6.0],    # bin 30-45
        [50.0, 7.0],    # bin 45+
        [60.0, 8.0],    # bin 45+
    ])
    y_pred = np.array([
        [7.0, 3.0],     # error: 2, 2
        [12.0, 4.0],    # error: 2, 2
        [23.0, 6.0],    # error: 3, 3
        [28.0, 7.0],    # error: 3, 3
        [40.0, 10.0],   # error: 5, 5
        [45.0, 11.0],   # error: 5, 5
        [58.0, 15.0],   # error: 8, 8
        [68.0, 16.0],   # error: 8, 8
    ])
    return y_pred, y_true


# ── Accuracy tests ──

class TestAccuracy:

    def test_perfect_predictions(self, perfect_predictions):
        y_pred, y_true = perfect_predictions
        result = get_accuracy(y_pred, y_true)
        assert result.mae_combined == 0.0
        assert result.mae_yaw == 0.0
        assert result.mae_pitch == 0.0

    def test_known_error(self, known_error_predictions):
        y_pred, y_true = known_error_predictions
        result = get_accuracy(y_pred, y_true)
        assert result.mae_combined == pytest.approx(2.0)
        assert result.mae_yaw == pytest.approx(2.0)
        assert result.mae_pitch == pytest.approx(2.0)

    def test_percentile_errors_shape(self, known_error_predictions):
        y_pred, y_true = known_error_predictions
        result = get_accuracy(y_pred, y_true)
        assert result.percentile_errors_yaw.shape == (3,)
        assert result.percentile_errors_pitch.shape == (3,)

    def test_percentile_errors_constant_error(self, known_error_predictions):
        """With constant error, all percentiles should equal the error."""
        y_pred, y_true = known_error_predictions
        result = get_accuracy(y_pred, y_true)
        np.testing.assert_allclose(result.percentile_errors_yaw, [2.0, 2.0, 2.0])
        np.testing.assert_allclose(result.percentile_errors_pitch, [2.0, 2.0, 2.0])

    def test_returns_accuracy_dataclass(self, known_error_predictions):
        y_pred, y_true = known_error_predictions
        result = get_accuracy(y_pred, y_true)
        assert isinstance(result, Accuracy)


# ── Safety tests ──

class TestSafety:

    def test_all_on_road(self):
        """All samples within threshold — no false positives or negatives."""
        y_true = np.array([[5.0, 3.0], [10.0, 5.0], [-8.0, 4.0]])
        y_pred = np.array([[6.0, 4.0], [9.0, 6.0], [-7.0, 3.0]])
        result = get_safety(y_pred, y_true, threshold_angle=15)
        assert result.fpr == 0.0
        assert result.fnr == 0.0

    def test_false_positive(self):
        """Model predicts off-road but ground truth is on-road."""
        y_true = np.array([[5.0, 3.0], [10.0, 5.0]])
        y_pred = np.array([[20.0, 3.0], [10.0, 5.0]])  # first pred exceeds threshold
        result = get_safety(y_pred, y_true, threshold_angle=15)
        assert result.fpr > 0.0
        assert result.fnr == 0.0

    def test_false_negative(self):
        """Ground truth is off-road but model predicts on-road."""
        y_true = np.array([[20.0, 3.0], [10.0, 5.0]])  # first true exceeds threshold
        y_pred = np.array([[5.0, 3.0], [10.0, 5.0]])
        result = get_safety(y_pred, y_true, threshold_angle=15)
        assert result.fpr == 0.0
        assert result.fnr > 0.0

    def test_pitch_triggers_off_road(self):
        """High pitch (looking down) should also count as off-road."""
        y_true = np.array([[5.0, 20.0], [5.0, 5.0]])
        y_pred = np.array([[5.0, 20.0], [5.0, 5.0]])
        result = get_safety(y_pred, y_true, threshold_angle=15)
        # both agree on classification, so no FP/FN
        assert result.fpr == 0.0
        assert result.fnr == 0.0

    def test_returns_safety_dataclass(self):
        y_true = np.array([[5.0, 3.0]])
        y_pred = np.array([[6.0, 4.0]])
        result = get_safety(y_pred, y_true, threshold_angle=15)
        assert isinstance(result, Safety)


# ── Slicing tests ──

class TestSlicing:

    def test_all_bins_populated(self, spread_yaw_data):
        y_pred, y_true = spread_yaw_data
        result = get_slicing(y_pred, y_true)
        assert isinstance(result.zero_to_fifteen, Accuracy)
        assert isinstance(result.fifteen_to_thirty, Accuracy)
        assert isinstance(result.thirty_to_forty_five, Accuracy)
        assert isinstance(result.forty_five_plus, Accuracy)

    def test_error_increases_with_yaw(self, spread_yaw_data):
        """Error should increase in higher yaw bins (by construction of fixture)."""
        y_pred, y_true = spread_yaw_data
        result = get_slicing(y_pred, y_true)
        assert result.zero_to_fifteen.mae_combined < result.fifteen_to_thirty.mae_combined
        assert result.fifteen_to_thirty.mae_combined < result.thirty_to_forty_five.mae_combined
        assert result.thirty_to_forty_five.mae_combined < result.forty_five_plus.mae_combined

    def test_returns_slicing_dataclass(self, spread_yaw_data):
        y_pred, y_true = spread_yaw_data
        result = get_slicing(y_pred, y_true)
        assert isinstance(result, Slicing)


# ── Report tests ──

class TestReport:

    def test_report_contains_all_sections(self, spread_yaw_data):
        y_pred, y_true = spread_yaw_data
        result = report(y_pred, y_true, threshold=15)
        assert isinstance(result, Report)
        assert isinstance(result.accuracy, Accuracy)
        assert isinstance(result.safety, Safety)
        assert isinstance(result.slicing, Slicing)

    def test_report_accuracy_matches_direct_call(self, spread_yaw_data):
        y_pred, y_true = spread_yaw_data
        result = report(y_pred, y_true)
        direct = get_accuracy(y_pred, y_true)
        assert result.accuracy.mae_combined == direct.mae_combined
