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
    # All floats should be converted to strings for JSON serialization clarity
    y_float = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    result = typify_classification(y_float)
    assert result.dtype.kind == 'U'  # Unicode string
    assert np.array_equal(result, ['0.0', '1.0', '0.0', '1.0'])


def test_typify_classification_float64_non_integers():
    """Test typify_classification with float64 values that are not integers"""
    # Test with non-integer float values - should fallback to string
    y_float = np.array([0.5, 1.5, 0.2], dtype=np.float64)
    result = typify_classification(y_float)
    assert result.dtype.kind == 'U'  # Unicode string
    assert np.array_equal(result, ['0.5', '1.5', '0.2'])


def test_typify_classification_edge_cases():
    """Test edge cases for typify_classification with floating-point values"""
    # Test with NaN values - should fall back to string
    y_with_nan = np.array([0.0, 1.0, np.nan], dtype=np.float64)
    result = typify_classification(y_with_nan)
    assert result.dtype.kind == 'U'  # Unicode string
    
    # Test with negative integer floats - should convert to strings
    y_negative = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    result = typify_classification(y_negative)
    assert result.dtype.kind == 'U'  # Unicode string
    assert np.array_equal(result, ['-1.0', '0.0', '1.0'])
    
    # Test with large integer floats - should convert to strings
    y_large = np.array([1e10, 2e10], dtype=np.float64)
    result = typify_classification(y_large)
    assert result.dtype.kind == 'U'  # Unicode string
    assert np.array_equal(result, ['10000000000.0', '20000000000.0'])


def test_gen_local_selector_with_string_ac_score():
    """Test gen_local_selector handles string AcScore values correctly"""
    from interpret.utils._explanation import gen_local_selector
    import numpy as np
    
    # Create test data that simulates string AcScore from float64 labels
    data_dicts = [
        {
            "perf": {
                "predicted": 1,
                "actual": 1,
                "predicted_score": 0.8,  # Always float from model predictions
                "actual_score": "0.0",   # String from typify_classification of float64 labels
            }
        },
        {
            "perf": {
                "predicted": 0,
                "actual": 0,
                "predicted_score": 0.3,
                "actual_score": "1.0",
            }
        }
    ]
    
    # This should not raise an error and should compute residuals correctly
    result = gen_local_selector(data_dicts, is_classification=True)
    
    # Check that residuals are computed correctly
    # For first record: float("0.0") - 0.8 = -0.8
    # For second record: float("1.0") - 0.3 = 0.7
    assert abs(result.iloc[0]["Resid"] - (-0.8)) < 1e-10
    assert abs(result.iloc[1]["Resid"] - 0.7) < 1e-10
    
    # Check that absolute residuals are computed correctly
    assert abs(result.iloc[0]["AbsResid"] - 0.8) < 1e-10
    assert abs(result.iloc[1]["AbsResid"] - 0.7) < 1e-10


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
