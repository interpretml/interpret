# Copyright (c) 2026 The InterpretML Contributors
# Distributed under the MIT software license

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from interpret.perf import PR, ROC
from interpret.utils._explanation import gen_perf_dicts
from interpret.utils._unify_predict import determine_classes


@pytest.mark.parametrize("label", [-5, 7, False, "only", "other"])
@pytest.mark.parametrize("n_samples", [1, 4])
def test_single_class_preserves_labels_and_prediction_scores(label, n_samples):
    X = np.arange(n_samples, dtype=float).reshape(-1, 1)
    y = np.full(n_samples, label)
    model = DummyClassifier(strategy="most_frequent").fit(X, y)
    original_classes = model.classes_.copy()

    with pytest.warns(UserWarning, match="single-class data"):
        predict_fn, n_classes, classes = determine_classes(model, X, n_samples)

    assert n_classes == 2
    assert classes[0] == model.classes_[0]
    assert classes[1] != classes[0]
    np.testing.assert_array_equal(model.classes_, original_classes)
    probabilities = predict_fn(X)
    np.testing.assert_array_equal(probabilities[:, 0], np.ones(n_samples))
    np.testing.assert_array_equal(probabilities[:, 1], np.zeros(n_samples))
    records = gen_perf_dicts(probabilities, y, True, classes)
    assert all(record["actual"] == 0 for record in records)
    assert all(record["actual_score"] == 1 for record in records)


@pytest.mark.parametrize("label", [-5, 7, False, "only", "other"])
@pytest.mark.parametrize("explainer_class", [PR, ROC])
def test_single_class_numeric_labels_work_in_performance_curves(label, explainer_class):
    X = np.arange(4, dtype=float).reshape(-1, 1)
    y = np.full(4, label)
    model = DummyClassifier(strategy="most_frequent").fit(X, y)

    # Single-class metrics may be undefined; label lookup must still succeed.
    with pytest.warns(UserWarning, match="single-class data"):
        explanation = explainer_class(model).explain_perf(X, y)

    np.testing.assert_array_equal(explanation.data()["scores"], np.zeros(4))


def test_binary_classifier_classes_and_probabilities_are_unchanged():
    X = np.arange(4, dtype=float).reshape(-1, 1)
    y = np.array([5, 5, 5, 9])
    model = DummyClassifier(strategy="prior").fit(X, y)
    predict_fn, n_classes, classes = determine_classes(model, X, len(X))

    assert n_classes == 2
    np.testing.assert_array_equal(classes, model.classes_)
    np.testing.assert_array_equal(predict_fn(X), model.predict_proba(X))
