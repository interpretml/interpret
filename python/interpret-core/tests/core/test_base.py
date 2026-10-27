# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import pytest
from interpret.core._sklearn import (
    _SKBaseEstimator,
    _SKClassifierMixin,
    _SKRegressorMixin,
)
from interpret.core.base import BaseExplanation, GlobalExplainer, LocalExplainer
from interpret.glassbox import (
    ClassificationTree,
    DecisionListClassifier,
    EBMClassifier,
    EBMRegressor,
    LinearRegression,
    LogisticRegression,
    RegressionTree,
)
from interpret.privacy import DPEBMClassifier, DPEBMRegressor


@pytest.mark.parametrize(
    ("estimator_class", "mixin", "estimator_type"),
    [
        (ClassificationTree, _SKClassifierMixin, "classifier"),
        (DecisionListClassifier, _SKClassifierMixin, "classifier"),
        (EBMClassifier, _SKClassifierMixin, "classifier"),
        (LogisticRegression, _SKClassifierMixin, "classifier"),
        (DPEBMClassifier, _SKClassifierMixin, "classifier"),
        (RegressionTree, _SKRegressorMixin, "regressor"),
        (EBMRegressor, _SKRegressorMixin, "regressor"),
        (LinearRegression, _SKRegressorMixin, "regressor"),
        (DPEBMRegressor, _SKRegressorMixin, "regressor"),
    ],
)
def test_estimator_inheritance(estimator_class, mixin, estimator_type):
    mro = estimator_class.__mro__
    assert mro.index(mixin) < mro.index(LocalExplainer)
    assert mro.index(LocalExplainer) < mro.index(GlobalExplainer)
    assert mro.index(GlobalExplainer) < mro.index(_SKBaseEstimator)
    tags = estimator_class().__sklearn_tags__()
    assert tags.estimator_type == estimator_type


def test_that_explanation_throws_exceptions_for_incomplete():
    class IncompleteExplanation(BaseExplanation):
        pass

    with pytest.raises(Exception):
        _ = IncompleteExplanation()


def test_that_explanation_works_for_complete():
    class CompleteExplanation(BaseExplanation):
        _internal_object = {"overall": None, "specific": [None]}
        explanation_type = "performance"
        selector = None
        name = ""

        def visualize(self, key=None):
            data_dict = self.data(key)
            # NOTE: Return a fig|df|text|dash-component
            return str(data_dict)

        def data(self, key=None):
            if key is None:
                return self._internal_object["overall"]
            return self._internal_object["specific"][key]

    try:
        explanation = CompleteExplanation()
        assert explanation.data() is None
        assert explanation.data(0) is None
        assert explanation.visualize(0) == str(None)
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")
