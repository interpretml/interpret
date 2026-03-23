# Copyright (c) 2024 The InterpretML Contributors
# Distributed under the MIT software license

import numpy as np
import pytest
import warnings
from aplr import APLRClassifier as APLRClassifierNative
from aplr import APLRRegressor as APLRRegressorNative
from interpret.glassbox import APLRClassifier, APLRRegressor
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.utils import estimator_checks


def test_regression():
    dataset = load_diabetes()
    X, y = dataset.data, dataset.target
    X = X[:100]
    y = y[:100]
    feature_names = dataset.feature_names

    native = APLRRegressorNative(max_interaction_level=2)
    our_aplr = APLRRegressor(max_interaction_level=2)

    native.fit(X, y, X_names=feature_names)
    our_aplr.fit(X, y, X_names=feature_names)

    native_pred = native.predict(X)
    our_pred = our_aplr.predict(X)
    assert np.allclose(native_pred, our_pred)

    # With response
    local_expl = our_aplr.explain_local(X[:5], y[:5])
    local_viz = local_expl.visualize(0)
    assert local_viz is not None

    # Without response
    local_expl = our_aplr.explain_local(X[:5])
    local_viz = local_expl.visualize(0)
    assert local_viz is not None

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Dropping term .* from explanation since we can't graph more than 2 dimensions.",
            category=UserWarning,
        )
        global_expl = our_aplr.explain_global()
        global_viz = global_expl.visualize()
        assert global_viz is not None


def test_regression_no_feature_names():
    dataset = load_diabetes()
    X, y = dataset.data, dataset.target
    X = X[:100]
    y = y[:100]

    our_aplr = APLRRegressor()
    our_aplr.fit(X, y)

    our_pred = our_aplr.predict(X)
    assert our_pred is not None
    assert our_aplr.feature_names_in_ == [f"X{i + 1}" for i in range(X.shape[1])]


def test_classification():
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    X = X[:100]
    y = y[:100]
    feature_names = cancer.feature_names
    y_native = (
        y.astype(str).tolist()
        if not all(isinstance(item, str) for item in y)
        else y.copy()
    )

    native = APLRClassifierNative(m=500, max_interaction_level=2, verbosity=1)
    our_aplr = APLRClassifier(m=500, max_interaction_level=2, verbosity=1)

    native.fit(X, y_native, X_names=feature_names)
    our_aplr.fit(X, y, X_names=feature_names)

    native_pred = native.predict_class_probabilities(X)
    our_pred = our_aplr.predict_class_probabilities(X)
    assert np.allclose(native_pred, our_pred)

    native_pred = native.predict(X)
    our_pred = our_aplr.predict(X)
    assert [str(v) for v in our_pred] == list(native_pred)

    # With response
    local_expl = our_aplr.explain_local(X[:5], y[:5])
    local_viz = local_expl.visualize(0)
    assert local_viz is not None

    # Without response
    local_expl = our_aplr.explain_local(X[:5])
    local_viz = local_expl.visualize(0)
    assert local_viz is not None

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Dropping term .* from explanation since we can't graph more than 2 dimensions.",
            category=UserWarning,
        )
        global_expl = our_aplr.explain_global()
        global_viz = global_expl.visualize()
        assert global_viz is not None


@pytest.fixture
def skip_sklearn() -> set:
    """Tests which we do not adhere to."""
    # TODO: whittle these down to the minimum
    return {
        "check_do_not_raise_errors_in_init_or_set_params",  # native APLR validates params eagerly in __init__/set_params
        "check_no_attributes_set_in_init",  # native APLR sets attributes in __init__
        "check_fit1d",  # interpret accepts 1d X for single feature
        "check_fit2d_predict1d",  # interpret accepts 1d for predict
        "check_supervised_y_2d",  # interpret deliberately supports y.shape = (nsamples, 1)
        "check_classifiers_regression_target",  # interpret is more permissive with y values
        "check_n_features_in_after_fitting",  # interpret uses a different error message format
        "check_complex_data",  # interpret uses a different error message for complex data
        "check_estimators_nan_inf",  # interpret treats NaN as missing data, not as NaN/inf validation error
        "check_requires_y_none",  # interpret uses a different error message for y=None
        # native APLR raises RuntimeError instead of ValueError for invalid inputs
        "check_regressors_train",  # native APLR raises RuntimeError for mismatched X/y lengths
        "check_regressor_data_not_an_array",  # native APLR raises RuntimeError for mismatched X/y lengths
        "check_classifier_data_not_an_array",  # native APLR raises RuntimeError for mismatched X/y lengths
        "check_classifiers_train",  # native APLR raises RuntimeError for mismatched X/y lengths
        "check_classifiers_classes",  # native APLR raises RuntimeError for mismatched X/y lengths
        "check_regressors_no_decision_function",  # native APLR raises RuntimeError for mismatched X/y lengths
        "check_supervised_y_no_nan",  # native APLR raises RuntimeError instead of ValueError for NaN y
        "check_estimators_empty_data_messages",  # native APLR raises RuntimeError for empty data
        "check_fit2d_1sample",  # native APLR requires more than 1 sample for CV folds
        # native APLR classifier-specific limitations
        "check_classifiers_one_label",  # native APLR requires at least 2 categories
        "check_classifiers_one_label_sample_weights",  # native APLR requires at least 2 categories
        "check_fit_idempotent",  # native APLR classifier fitting twice produces different results
        "check_sample_weight_equivalence_on_dense_data",  # algorithmic difference
        "check_sample_weight_equivalence_on_sparse_data",  # algorithmic difference
    }


@estimator_checks.parametrize_with_checks(
    [
        APLRRegressor(cv_folds=2),
        APLRClassifier(cv_folds=2),
    ]
)
def test_sklearn_estimator(estimator, check, skip_sklearn):
    if check.func.__name__ in skip_sklearn:
        pytest.skip("Deliberate deviation from scikit-learn.")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            "Casting complex values to real discards the imaginary part",
            category=np.exceptions.ComplexWarning,
        )
        check(estimator)
