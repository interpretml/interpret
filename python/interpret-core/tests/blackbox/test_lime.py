# Copyright (c) 2026 The InterpretML Contributors
# Distributed under the MIT software license

"""Tests for LimeTabular wrapper.

Issue #477: passing ``mode="classification"`` to ``LimeTabular`` was
silently overridden to ``"regression"`` because the wrapper unifies
classifier and regressor predict functions to a scalar. The override is
intentional (LIME wouldn't otherwise work with the unified predict
path), but a silent override is surprising — users were left wondering
why the keyword had no effect. The wrapper now emits a UserWarning when
a non-default ``mode`` is discarded.
"""

import warnings

import numpy as np
import pytest

pytest.importorskip("lime")

from interpret.blackbox import LimeTabular


def _toy_data():
    rng = np.random.default_rng(0)
    return rng.standard_normal((20, 3)).astype(np.float64)


def _toy_predict_fn(X):
    # Probability of class 1 — what LimeTabular's unify_predict_fn yields
    # for a binary classifier.
    return 1.0 / (1.0 + np.exp(-X.sum(axis=1)))


def test_user_supplied_mode_classification_warns():
    # BEFORE: passing mode="classification" was silently overridden.
    # AFTER: a UserWarning is emitted that mentions the override.
    data = _toy_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        LimeTabular(
            model=_toy_predict_fn,
            data=data,
            mode="classification",
        )
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1, [str(w.message) for w in caught]
    assert "regression" in str(user_warnings[0].message).lower()
    assert "classification" in str(user_warnings[0].message).lower()


def test_no_warning_when_mode_omitted():
    # The default path (no mode kwarg) must stay quiet — we don't want to
    # nag every user who follows the documented API.
    data = _toy_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        LimeTabular(model=_toy_predict_fn, data=data)
    user_warnings = [
        w
        for w in caught
        if issubclass(w.category, UserWarning) and "mode=" in str(w.message)
    ]
    assert user_warnings == []


def test_no_warning_when_mode_regression():
    # Explicitly passing the value we use internally must also stay quiet.
    data = _toy_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        LimeTabular(model=_toy_predict_fn, data=data, mode="regression")
    user_warnings = [
        w
        for w in caught
        if issubclass(w.category, UserWarning) and "mode=" in str(w.message)
    ]
    assert user_warnings == []
