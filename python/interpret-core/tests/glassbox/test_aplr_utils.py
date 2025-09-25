# Copyright (c) 2024 The InterpretML Contributors
# Distributed under the MIT software license

import numpy as np
import pandas as pd

from interpret.glassbox._aplr import (
    convert_to_numpy_matrix,
    define_feature_names,
)


def test_convert_to_numpy_matrix_success():
    # Test with numpy array
    X_np = np.array([[1, 2], [3, 4]])
    assert np.array_equal(convert_to_numpy_matrix(X_np), X_np.astype(np.float64))

    # Test with pandas DataFrame
    X_pd = pd.DataFrame([[1, 2], [3, 4]])
    assert np.array_equal(convert_to_numpy_matrix(X_pd), X_pd.astype(np.float64).values)

    # Test with list of lists
    X_list = [[1, 2], [3, 4]]
    assert np.array_equal(
        convert_to_numpy_matrix(X_list), np.array(X_list, dtype=np.float64)
    )


def test_convert_to_numpy_matrix_failures():
    # Test with non-numeric numpy array
    X_np_non_numeric = np.array([["a", "b"], ["c", "d"]])
    try:
        convert_to_numpy_matrix(X_np_non_numeric)
        assert False, "Expected TypeError for non-numeric numpy array"
    except TypeError as e:
        assert "must contain only numeric values" in str(e)

    # Test with non-numeric pandas DataFrame
    X_pd_non_numeric = pd.DataFrame([["a", 2], ["c", 4]])
    try:
        convert_to_numpy_matrix(X_pd_non_numeric)
        assert False, "Expected TypeError for non-numeric pandas DataFrame"
    except TypeError as e:
        assert "all columns must be numeric" in str(e)

    # Test with non-numeric list of lists
    X_list_non_numeric = [[1, "b"], [3, 4]]
    try:
        convert_to_numpy_matrix(X_list_non_numeric)
        assert False, "Expected TypeError for non-numeric list of lists"
    except TypeError as e:
        assert "must be a list of lists" in str(e)

    # Test with unsupported type
    try:
        convert_to_numpy_matrix("unsupported_type")
        assert False, "Expected TypeError for unsupported type"
    except TypeError as e:
        assert "must be a numpy matrix" in str(e)


def test_define_feature_names_with_names():
    X = np.array([[1, 2], [3, 4]])
    X_names = ["feature1", "feature2"]
    assert define_feature_names(X, X_names=X_names) == X_names


def test_define_feature_names_without_names():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    assert define_feature_names(X) == ["X1", "X2", "X3"]


def test_define_feature_names_empty_names():
    X = np.array([[1, 2], [3, 4]])
    assert define_feature_names(X, X_names=[]) == ["X1", "X2"]


def test_create_values():
    from interpret.glassbox._aplr import create_values

    X = np.array([[10, 20], [30, 40]])
    explanations = np.zeros((2, 3))
    term_names = ["feature1", "interaction", "feature2"]
    feature_names = ["feature1", "feature2"]
    X_values = create_values(X, explanations, term_names, feature_names)
    assert np.array_equal(X_values[:, 0], [10, 30])
    assert np.isnan(X_values[:, 1]).all()
    assert np.array_equal(X_values[:, 2], [20, 40])
