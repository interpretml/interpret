# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import numpy as np
import pytest
from interpret.blackbox._partialdependence import PartialDependence, _gen_pdp
from sklearn.linear_model import LinearRegression


def _make_regression_data(n_samples):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, 2))
    y = X[:, 0] * 2.0 - X[:, 1] + 0.5
    return X, y


@pytest.mark.parametrize("n_samples", [1, 2, 5, 9])
def test_gen_pdp_fewer_rows_than_ice_samples(n_samples):
    """Datasets with fewer rows than the default num_ice_samples must not crash.

    Before the fix, sampling num_ice_samples (10) rows without replacement
    raised "ValueError: Cannot take a larger sample than population when
    'replace=False'" for any dataset with fewer than 10 rows.
    """
    X, y = _make_regression_data(n_samples)
    model = LinearRegression().fit(X, y)

    pdp = _gen_pdp(
        X,
        model.predict,
        col_idx=0,
        feature_type="continuous",
        num_points=10,
        std_coef=1.0,
    )

    # One background line per available row, no more.
    assert pdp["background_scores"].shape[0] == n_samples
    assert pdp["background_scores"].shape[1] == len(pdp["names"])


def test_partial_dependence_small_dataset_does_not_crash():
    """PartialDependence must work end-to-end on a dataset with < 10 rows."""
    X, y = _make_regression_data(3)
    model = LinearRegression().fit(X, y)

    pdp = PartialDependence(
        model, X, feature_names=["a", "b"], feature_types=["continuous", "continuous"]
    )

    assert len(pdp.pdps_) == 2
    for pdp_item in pdp.pdps_:
        assert pdp_item["background_scores"].shape[0] == 3

    explanation = pdp.explain_global()
    assert explanation.feature_names == ["a", "b"]


def test_gen_pdp_enough_rows_still_caps_at_num_ice_samples():
    """With enough rows, exactly num_ice_samples background lines are kept."""
    X, y = _make_regression_data(25)
    model = LinearRegression().fit(X, y)

    pdp = _gen_pdp(
        X,
        model.predict,
        col_idx=0,
        feature_type="continuous",
        num_points=10,
        std_coef=1.0,
        num_ice_samples=10,
    )

    assert pdp["background_scores"].shape[0] == 10
