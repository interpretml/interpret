# Copyright (c) 2026 The InterpretML Contributors
# Distributed under the MIT software license

"""Regression tests for Marginal.visualize() on categorical features.

Issue #119: ``Marginal(...).explain_data(...).visualize(key)`` raised
``TypeError: unsupported operand type(s) for -: 'str' and 'str'`` when
``feature_types[key]`` was ``"nominal"`` because the histogram path
unconditionally computed ``density_dict["names"][1] - density_dict["names"][0]``
even though categorical features carry string labels in ``names``.
"""

import numpy as np
import pandas as pd
import pytest
from interpret.data import Marginal


@pytest.fixture
def mixed_dataframe():
    return pd.DataFrame(
        {
            "cat": ["a", "b", "a", "c", "b", "a", "c", "b"],
            "cont": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )


def test_visualize_nominal_feature_continuous_target_succeeds(mixed_dataframe):
    # BEFORE: this call raised TypeError on str-str subtraction.
    # AFTER: returns a plotly Figure.
    y = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
    explanation = Marginal(feature_types=["nominal", "continuous"]).explain_data(
        mixed_dataframe, y
    )
    fig = explanation.visualize(0)
    assert fig is not None
    # We don't depend on plotly types beyond the .data attribute existing.
    assert hasattr(fig, "data")


def test_visualize_continuous_feature_still_works(mixed_dataframe):
    # Non-regression: the continuous branch must continue to work after
    # the categorical fix.
    y = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
    explanation = Marginal(feature_types=["nominal", "continuous"]).explain_data(
        mixed_dataframe, y
    )
    fig = explanation.visualize(1)
    assert fig is not None
    assert hasattr(fig, "data")


def test_visualize_nominal_feature_classification_target(mixed_dataframe):
    # Classification target: response_density["names"] are also categorical
    # (class labels). The fix must handle that branch too.
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    explanation = Marginal(feature_types=["nominal", "continuous"]).explain_data(
        mixed_dataframe, y
    )
    fig = explanation.visualize(0)
    assert fig is not None


def test_visualize_overall_unaffected(mixed_dataframe):
    # The key=None path is independent of the bug; guard against
    # accidental regression.
    y = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
    explanation = Marginal(feature_types=["nominal", "continuous"]).explain_data(
        mixed_dataframe, y
    )
    fig = explanation.visualize(None)
    assert fig is not None
