# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import numpy as np
from interpret.utils._clean_simple import clean_dimensions, typify_classification


def test_clean_dimensions_2d():
    init_score = [
        [[[((x,) for x in [1, 2])]]],
        [3, 4],
        (np.array([5, 6]),),
        np.array([[[(7,), (8,)]]]),
    ]
    init_score = clean_dimensions(init_score, "init_score")
    init_score = init_score.astype(np.float64, copy=False)
    assert init_score.shape == (4, 2)
    assert init_score[0, 0] == 1
    assert init_score[0, 1] == 2
    assert init_score[1, 0] == 3
    assert init_score[1, 1] == 4
    assert init_score[2, 0] == 5
    assert init_score[2, 1] == 6
    assert init_score[3, 0] == 7
    assert init_score[3, 1] == 8


def test_typify_classification_float64_integers():
    """Test typify_classification with float64 values that are integers (e.g., 0.0, 1.0)"""
    # Test binary classification with float64 labels
    y_float = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    result = typify_classification(y_float)
    assert result.dtype == np.int64
    assert np.array_equal(result, [0, 1, 0, 1])


def test_typify_classification_float64_non_integers():
    """Test typify_classification with float64 values that are not integers"""
    # Test with non-integer float values - should fallback to string
    y_float = np.array([0.5, 1.5, 0.2], dtype=np.float64)
    result = typify_classification(y_float)
    assert result.dtype.kind == 'U'  # Unicode string
    assert np.array_equal(result, ['0.5', '1.5', '0.2'])


def test_typify_classification_existing_types():
    """Test typify_classification with existing supported types"""
    # Test integers
    y_int = np.array([0, 1, 0, 1], dtype=np.int32)
    result = typify_classification(y_int)
    assert result.dtype == np.int64
    assert np.array_equal(result, [0, 1, 0, 1])
    
    # Test booleans
    y_bool = np.array([True, False, True], dtype=np.bool_)
    result = typify_classification(y_bool)
    assert result.dtype == np.bool_
    assert np.array_equal(result, [True, False, True])
