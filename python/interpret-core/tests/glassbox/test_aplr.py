# Copyright (c) 2024 The InterpretML Contributors
# Distributed under the MIT software license

import numpy as np
from aplr import APLRClassifier as APLRClassifierNative
from aplr import APLRRegressor as APLRRegressorNative
from interpret.glassbox import APLRClassifier, APLRRegressor
from sklearn.datasets import load_breast_cancer, load_diabetes
import warnings


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
    assert our_aplr.feature_names_in_ == [f"X{i+1}" for i in range(X.shape[1])]


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
    assert native_pred == our_pred

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