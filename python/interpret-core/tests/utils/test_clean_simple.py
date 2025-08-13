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


def test_shap_kernel_float64_classification_labels():
    """Test ShapKernel with float64 classification labels (reproduces issue #609)"""
    try:
        import shap
        from sklearn.ensemble import RandomForestClassifier
        from interpret.blackbox import ShapKernel
        import pandas as pd
    except ImportError:
        # Skip test if shap or sklearn not available
        import pytest
        pytest.skip("SHAP or sklearn not available")
    
    # Create synthetic data for testing
    np.random.seed(42)
    X_train = np.random.randn(100, 4)
    X_test = np.random.randn(5, 4)
    
    # Use float64 classification labels that would previously cause UFuncTypeError
    y_train_float64 = np.array([0.0, 1.0] * 50, dtype=np.float64)
    y_test_float64 = np.array([0.0, 1.0, 0.0, 1.0, 1.0], dtype=np.float64)
    
    # Train a model with float64 labels
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train_float64)
    
    # This should not raise UFuncTypeError anymore
    shap_kernel = ShapKernel(model, X_train[:20], feature_names=['A', 'B', 'C', 'D'])
    explanation = shap_kernel.explain_local(X_test[:1], y_test_float64[:1])
    
    # Verify the explanation is properly created
    assert explanation is not None
    assert hasattr(explanation, 'data')
    assert hasattr(explanation, 'visualize')
    
    # Verify the explanation has expected structure
    local_data = explanation.data(0)
    assert isinstance(local_data, dict)
    assert 'scores' in local_data
    assert 'names' in local_data
