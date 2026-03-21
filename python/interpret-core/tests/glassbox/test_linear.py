# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import numpy as np
from interpret.glassbox import LinearRegression, LogisticRegression
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.linear_model import LinearRegression as SKLinear
from sklearn.linear_model import LogisticRegression as SKLogistic
from sklearn.utils import estimator_checks
import pytest
import warnings


def test_linear_regression():
    dataset = load_diabetes()
    X, y = dataset.data, dataset.target
    feature_names = dataset.feature_names

    sk_lr = SKLinear()
    our_lr = LinearRegression(feature_names=feature_names)

    sk_lr.fit(X, y)
    our_lr.fit(X, y)

    sk_pred = sk_lr.predict(X)
    our_pred = our_lr.predict(X)
    assert np.allclose(sk_pred, our_pred)

    # With labels
    local_expl = our_lr.explain_local(X, y)
    local_viz = local_expl.visualize(0)
    assert local_viz is not None

    # Without labels
    local_expl = our_lr.explain_local(X)
    local_viz = local_expl.visualize(0)
    assert local_viz is not None

    global_expl = our_lr.explain_global()
    global_viz = global_expl.visualize()
    assert global_viz is not None


def test_logistic_regression():
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    feature_names = cancer.feature_names

    sk_lr = SKLogistic(tol=0.01, random_state=1)
    our_lr = LogisticRegression(tol=0.01, feature_names=feature_names, random_state=1)

    sk_lr.fit(X, y)
    our_lr.fit(X, y)

    sk_pred = sk_lr.predict_proba(X)
    our_pred = our_lr.predict_proba(X)
    assert np.allclose(sk_pred, our_pred)

    sk_pred = sk_lr.predict(X)
    our_pred = our_lr.predict(X)
    assert np.allclose(sk_pred, our_pred)

    # With labels
    local_expl = our_lr.explain_local(X, y)
    local_viz = local_expl.visualize(0)
    assert local_viz is not None

    # Without labels
    local_expl = our_lr.explain_local(X)
    local_viz = local_expl.visualize(0)
    assert local_viz is not None

    global_expl = our_lr.explain_global()
    global_viz = global_expl.visualize()
    assert global_viz is not None


def test_sorting():
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    feature_names = cancer.feature_names

    sk_lr = SKLogistic(tol=0.01, random_state=1)
    our_lr = LogisticRegression(tol=0.01, feature_names=feature_names, random_state=1)

    sk_lr.fit(X, y)
    our_lr.fit(X, y)

    sk_pred = sk_lr.predict_proba(X)
    our_pred = our_lr.predict_proba(X)
    assert np.allclose(sk_pred, our_pred)

    sk_pred = sk_lr.predict(X)
    our_pred = our_lr.predict(X)
    assert np.allclose(sk_pred, our_pred)

    # With labels
    local_expl = our_lr.explain_local(X, y)
    local_viz = local_expl.visualize(0)
    assert local_viz is not None

    # Without labels
    local_expl = our_lr.explain_local(X)
    local_viz = local_expl.visualize(0)
    assert local_viz is not None

    global_expl = our_lr.explain_global()
    global_viz = global_expl.visualize()
    assert global_viz is not None

    from interpret.visual.plot import get_sort_indexes, mli_sort_take, sort_take

    data_dict = sort_take(
        global_expl.data(), sort_fn=lambda x: -abs(x), top_n=15, reverse_results=True
    )
    scores_data = global_expl.data(-1)["mli"][0]["value"]["scores"]
    sort_indexes = get_sort_indexes(scores_data, sort_fn=lambda x: -abs(x), top_n=15)
    sorted_scores = mli_sort_take(scores_data, sort_indexes, reverse_results=True)
    assert data_dict["scores"] == sorted_scores


@pytest.fixture
def skip_sklearn() -> set:
    """Test which we do not adhere to."""
    # TODO: whittle these down to the minimum
    return {
        "check_do_not_raise_errors_in_init_or_set_params",  # LinearRegression accepts **kwargs for the underlying sklearn model
        "check_fit1d",  # interpret accepts 1d X for single feature
        "check_fit2d_predict1d",  # interpret accepts 1d for predict
        "check_supervised_y_2d",  # interpret deliberately supports y.shape = (nsamples, 1)
        "check_classifiers_regression_target",  # interpret is more permissive with y values
        "check_n_features_in_after_fitting",  # interpret uses a different error message format
        "check_complex_data",  # interpret uses a different error message for complex data
        "check_estimators_nan_inf",  # interpret treats NaN as missing data, not as NaN/inf validation error
        "check_requires_y_none",  # interpret uses a different error message for y=None
    }


@estimator_checks.parametrize_with_checks(
    [
        LinearRegression(),
        LogisticRegression(),
    ]
)
def test_sklearn_estimator(estimator, check, skip_sklearn):
    if check.func.__name__ in skip_sklearn:
        pytest.skip("Deliberate deviation from scikit-learn.")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            "Detected multiclass problem. Forcing interactions to 0.",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            "Casting complex values to real discards the imaginary part",
            category=np.exceptions.ComplexWarning,
        )
        check(estimator)
