# Copyright (c) 2026 The InterpretML Contributors
# Distributed under the MIT software license

import numpy as np
import pytest
from interpret.perf import CalibrationCurve
from sklearn.base import BaseEstimator, ClassifierMixin


class BinaryProbabilityModel(ClassifierMixin, BaseEstimator):
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        scores = np.asarray(X)[:, 0]
        return np.column_stack((1.0 - scores, scores))


def test_calibration_curve_data():
    X = np.array([[0.1], [0.2], [0.8], [0.9]])
    y = np.array([0, 0, 1, 1])

    explanation = CalibrationCurve(
        BinaryProbabilityModel(),
        n_bins=2,
    ).explain_perf(X, y)
    data = explanation.data()

    np.testing.assert_allclose(data["x_values"], [0.15, 0.85])
    np.testing.assert_allclose(data["y_values"], [0.0, 1.0])
    np.testing.assert_array_equal(data["density"]["scores"], [2, 2])
    np.testing.assert_allclose(data["density"]["names"], [0.0, 0.5, 1.0])
    assert data["n_bins"] == 2
    assert data["strategy"] == "uniform"


def test_calibration_curve_quantile_strategy():
    X = np.array([[0.1], [0.2], [0.3], [0.9]])
    y = np.array([0, 0, 1, 1])

    explanation = CalibrationCurve(
        BinaryProbabilityModel(),
        n_bins=2,
        strategy="quantile",
    ).explain_perf(X, y)
    data = explanation.data()

    np.testing.assert_allclose(data["x_values"], [0.15, 0.6])
    np.testing.assert_allclose(data["y_values"], [0.0, 1.0])
    assert data["strategy"] == "quantile"


def test_calibration_curve_visualize():
    X = np.array([[0.1], [0.2], [0.8], [0.9]])
    y = np.array([0, 0, 1, 1])

    explanation = CalibrationCurve(
        BinaryProbabilityModel(),
        n_bins=2,
    ).explain_perf(X, y, name="Test model")
    figure = explanation.visualize()

    assert len(figure.data) == 3
    assert figure.data[0].mode == "lines+markers"
    assert figure.data[0].name == "Calibration"
    assert figure.data[1].name == "Perfect calibration"
    assert figure.layout.title.text == "Calibration Curve: Test model"


@pytest.mark.parametrize("n_bins", [0, -1, 1.5, True])
def test_calibration_curve_rejects_invalid_n_bins(n_bins):
    with pytest.raises(ValueError, match="positive integer"):
        CalibrationCurve(BinaryProbabilityModel(), n_bins=n_bins)


def test_calibration_curve_rejects_invalid_strategy():
    with pytest.raises(ValueError, match="uniform"):
        CalibrationCurve(BinaryProbabilityModel(), strategy="invalid")
