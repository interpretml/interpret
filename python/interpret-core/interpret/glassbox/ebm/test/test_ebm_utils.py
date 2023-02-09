from math import ceil, floor
from ..utils import EBMUtils, _convert_categorical_to_continuous, _create_proportional_tensor
from ....utils import unify_data, unify_vector
from ....test.utils import (
    synthetic_regression,
    adult_classification
)

import numpy as np
import pytest

@pytest.mark.skip(reason="make_bag test needs to be updated")
def test_make_bag_regression():
    data = synthetic_regression()

    X_orig = data["full"]["X"]
    y_orig = data["full"]["y"]

    X, y, _, _ = unify_data(X_orig, y_orig)

    w = np.ones_like(y, dtype=np.float64)
    w = unify_vector(w).astype(np.float64, casting="unsafe", copy=False)

    test_size = 0.20

    X_train, X_val, y_train, y_val, w_train, w_val, _, _ = EBMUtils.make_bag(
        X,
        y,
        w,
        test_size=test_size,
        random_state=1,
        is_classification=False
    )

    num_samples = X.shape[0]
    num_features = X.shape[1]
    num_test_expected = ceil(test_size * num_samples)
    num_train_expected = num_samples - num_test_expected

    assert X_train.shape == (num_features, num_train_expected)
    assert X_val.shape == (num_features, num_test_expected)
    assert y_train.shape == (num_train_expected, )
    assert y_val.shape == (num_test_expected, )
    assert w_train.shape == (num_train_expected, )
    assert w_val.shape == (num_test_expected, )

    X_all = np.concatenate((X_train.T, X_val.T))
    np.array_equal(np.sort(X, axis=0), np.sort(X_all, axis=0))

@pytest.mark.skip(reason="make_bag test needs to be updated")
def test_make_bag_classification():
    data = adult_classification()

    X_orig = data["full"]["X"]
    y_orig = data["full"]["y"]

    X, y, _, _ = unify_data(X_orig, y_orig)

    w = np.ones_like(y, dtype=np.float64)
    w = unify_vector(w).astype(np.float64, casting="unsafe", copy=False)

    test_size = 0.20

    X_train, X_val, y_train, y_val, w_train, w_val, _, _ = EBMUtils.make_bag(
        X,
        y,
        w,
        test_size=test_size,
        random_state=1,
        is_classification=True
    )

    num_samples = X.shape[0]
    num_features = X.shape[1]
    num_test_expected = ceil(test_size * num_samples)
    num_train_expected = num_samples - num_test_expected

    # global guarantee: correct number of overall train/val/weights returned
    assert X_train.shape == (num_features, num_train_expected)
    assert X_val.shape == (num_features, num_test_expected)
    assert y_train.shape == (num_train_expected, )
    assert y_val.shape == (num_test_expected, )
    assert w_train.shape == (num_train_expected, )
    assert w_val.shape == (num_test_expected, )

    X_all = np.concatenate((X_train.T, X_val.T))
    np.array_equal(np.sort(X, axis=0), np.sort(X_all, axis=0))

    # per class guarantee: train/val count should be no more than one away from ideal
    class_counts = np.bincount(y)
    train_class_counts = np.bincount(y_train)
    val_class_counts = np.bincount(y_val)
    ideal_training = num_train_expected / num_samples
    ideal_val = num_test_expected / num_samples
    for label in set(y):
        ideal_training_count = ideal_training * class_counts[label]
        ideal_val_count = ideal_val * class_counts[label]

        assert (train_class_counts[label] == ceil(ideal_training_count) 
            or train_class_counts[label] == floor(ideal_training_count) 
            or train_class_counts[label] == ideal_training_count)
        assert (val_class_counts[label] == ceil(ideal_val_count) 
            or val_class_counts[label] == floor(ideal_val_count) 
            or val_class_counts[label] == ideal_val_count)

def test_convert_categorical_to_continuous_easy():
    cuts, mapping, old_min, old_max = _convert_categorical_to_continuous({"10": 1, "20": 2, "30": 3})
    assert len(cuts) == 2
    assert cuts[0] == 15
    assert cuts[1] == 25
    assert(mapping == [[0], [1], [2], [3], [4]])
    assert old_min == 10
    assert old_max == 30

def test_convert_categorical_to_continuous_overlap():
    cuts, mapping, old_min, old_max = _convert_categorical_to_continuous({"10": 1, "+5": 1, "40": 4, "abc": 2, "20": 2, "25": 1, "30": 3, "35": 3})
    assert len(cuts) == 2
    assert cuts[0] == 27.5
    assert cuts[1] == 37.5
    assert(mapping == [[0], [1, 2], [3], [4], [5]])
    assert old_min == 5
    assert old_max == 40

def test_convert_categorical_to_continuous_identical():
    cuts, mapping, old_min, old_max = _convert_categorical_to_continuous({"10": 1, "+20": 2, "  20  ": 3, "30": 4})
    assert len(cuts) == 2
    assert cuts[0] == 15
    assert cuts[1] == 25
    assert(mapping == [[0], [1], [2, 3], [4], [5]])
    assert old_min == 10
    assert old_max == 30

def test_create_proportional_tensor():
    axis_weights = [np.array([1, 2], np.float64), np.array([5, 15, 7], np.float64)]
    tensor = _create_proportional_tensor(axis_weights)
    # geometric mean is 9, so each should sum to that
    expected = np.array([
       [0.55555556, 1.66666667, 0.77777778],
       [1.11111111, 3.33333333, 1.55555556]
    ])
    assert np.allclose(tensor, expected)
