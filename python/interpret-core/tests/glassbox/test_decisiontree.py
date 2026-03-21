# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import numpy as np
from interpret.glassbox import ClassificationTree, RegressionTree
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.tree import DecisionTreeClassifier as SKDT
from sklearn.tree import DecisionTreeRegressor as SKRT
from sklearn.utils import estimator_checks
import pytest
import warnings


def test_rt():
    dataset = load_diabetes()
    X, y = dataset.data, dataset.target
    feature_names = dataset.feature_names

    sk_dt = SKRT(random_state=1, max_depth=3)
    our_dt = RegressionTree(feature_names=feature_names, random_state=1)

    sk_dt.fit(X, y)
    our_dt.fit(X, y)

    sk_pred = sk_dt.predict(X)
    our_pred = our_dt.predict(X)
    assert np.allclose(sk_pred, our_pred)

    # With labels
    local_expl = our_dt.explain_local(X, y)
    local_viz = local_expl.visualize(0)
    assert local_viz is not None

    # Without labels
    local_expl = our_dt.explain_local(X)
    local_viz = local_expl.visualize(0)
    assert local_viz is not None

    global_expl = our_dt.explain_global()
    global_viz = global_expl.visualize()
    assert global_viz is not None


def test_dt():
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    feature_names = cancer.feature_names

    sk_dt = SKDT(random_state=1, max_depth=3)
    our_dt = ClassificationTree(feature_names=feature_names, random_state=1)

    sk_dt.fit(X, y)
    our_dt.fit(X, y)

    sk_pred = sk_dt.predict_proba(X)
    our_pred = our_dt.predict_proba(X)
    assert np.allclose(sk_pred, our_pred)

    sk_pred = sk_dt.predict(X)
    our_pred = our_dt.predict(X)
    assert np.allclose(sk_pred, our_pred)

    # With labels
    local_expl = our_dt.explain_local(X, y)
    local_viz = local_expl.visualize(0)
    assert local_viz is not None

    # Without labels
    local_expl = our_dt.explain_local(X)
    local_viz = local_expl.visualize(0)
    assert local_viz is not None

    global_expl = our_dt.explain_global()
    global_viz = global_expl.visualize()
    assert global_viz is not None


@pytest.fixture
def skip_sklearn() -> set:
    """Test which we do not adhere to."""
    # TODO: whittle these down to the minimum
    return {
        "check_do_not_raise_errors_in_init_or_set_params",  # kwargs not a settable param
        "check_fit1d",  # we accept 1D X as a single-feature input
        "check_fit2d_predict1d",  # we accept 1D X at predict time
        "check_supervised_y_2d",  # we don't emit DataConversionWarning for 2D y
        "check_no_attributes_set_in_init",  # we store **kwargs for forwarding to sklearn
        "check_sample_weight_equivalence_on_dense_data",  # algorithmic difference
        "check_sample_weight_equivalence_on_sparse_data",  # algorithmic difference
        "check_classifiers_one_label",  # returns string class labels
        "check_classifiers_regression_target",  # we don't reject continuous targets
    }


@estimator_checks.parametrize_with_checks(
    [
        RegressionTree(),
        ClassificationTree(),
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
