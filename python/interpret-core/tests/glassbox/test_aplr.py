# Copyright (c) 2024 The InterpretML Contributors
# Distributed under the MIT software license

from aplr import APLRRegressor as APLRRegressorNative
from aplr import APLRClassifier as APLRClassifierNative
from interpret.glassbox._aplr import APLRRegressor, APLRClassifier
from sklearn.datasets import load_breast_cancer, load_diabetes
import numpy as np


def test_regression():
    dataset = load_diabetes()
    X, y = dataset.data, dataset.target
    feature_names = dataset.feature_names

    native = APLRRegressorNative(max_interaction_level=2)
    our_aplr = APLRRegressor(max_interaction_level=2)

    native.fit(X, y, X_names=feature_names)
    our_aplr.fit(X, y, X_names=feature_names)

    native_pred = native.predict(X)
    our_pred = our_aplr.predict(X)
    assert np.allclose(native_pred, our_pred)

    # # With response
    # local_expl = our_aplr.explain_local(X, y)
    # local_viz = local_expl.visualize(0)
    # assert local_viz is not None

    # # Without response
    # local_expl = our_aplr.explain_local(X)
    # local_viz = local_expl.visualize(0)
    # assert local_viz is not None

    global_expl = our_aplr.explain_global()
    global_viz = global_expl.visualize()
    assert global_viz is not None


def test_classification():
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    feature_names = cancer.feature_names

    native = APLRClassifierNative()
    our_aplr = APLRClassifier()

    native.fit(X, y, X_names=feature_names)
    our_aplr.fit(X, y, X_names=feature_names)

    native_pred = native.predict_class_probabilities(X)
    our_pred = our_aplr.predict_class_probabilities(X)
    assert np.allclose(native_pred, our_pred)

    native_pred = native.predict(X)
    our_pred = our_aplr.predict(X)
    assert np.allclose(native_pred, our_pred)

    # With labels
    local_expl = our_aplr.explain_local(X, y)
    local_viz = local_expl.visualize(0)
    assert local_viz is not None

    # Without labels
    local_expl = our_aplr.explain_local(X)
    local_viz = local_expl.visualize(0)
    assert local_viz is not None

    global_expl = our_aplr.explain_global()
    global_viz = global_expl.visualize()
    assert global_viz is not None


if __name__ == "__main__":
    test_regression()
    test_classification()
