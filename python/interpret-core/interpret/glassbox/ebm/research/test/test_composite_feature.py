import pandas as pd
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from ...ebm import ExplainableBoostingRegressor, ExplainableBoostingClassifier
from .....test.utils import synthetic_multiclass, synthetic_classification, synthetic_regression
from ..composite_feature import (
    compute_composite_feature_importance,
    _get_composite_feature_name,
    append_composite_feature_importance
)

def test_composite_feature_name():
    mocked_ebm_feat_names = ["Ft1", "Ft2", "Ft3", "Ft4", "Ft1 x Ft2"]

    assert "Ft3" == _get_composite_feature_name(["Ft3"], mocked_ebm_feat_names)

    composite_names = ["Ft1", "Ft3", "Ft1 x Ft2"]
    assert "Ft1 and Ft3 and Ft1 x Ft2" == _get_composite_feature_name(composite_names, mocked_ebm_feat_names)

    # Ft2, Ft4, Ft1 x Ft2
    composite_indices = [1, 3, 4]
    assert "Ft2 and Ft4 and Ft1 x Ft2" == _get_composite_feature_name(composite_indices, mocked_ebm_feat_names)

    out_of_bound_indices = [-1, 5]
    assert "" == _get_composite_feature_name(out_of_bound_indices, mocked_ebm_feat_names)

def test_append_composite_feature_importance():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]
    composite_names = ["A", "B"]

    ebm = ExplainableBoostingRegressor()
    ebm.fit(X, y)
    global_explanation = ebm.explain_global()
    local_explanation = ebm.explain_local(X)

    # An exception should be raised when the EBM is not fitted
    non_fitted_ebm = ExplainableBoostingRegressor()
    with pytest.raises(NotFittedError):
        append_composite_feature_importance(composite_names, non_fitted_ebm, global_explanation, X)

    # An exception should be raised when the explanation is not valid
    with pytest.raises(ValueError):
        append_composite_feature_importance(composite_names, ebm, local_explanation, X)

    wrong_global_exp = ebm.explain_global()
    wrong_global_exp._internal_obj = None
    with pytest.raises(ValueError):
        append_composite_feature_importance(composite_names, ebm, wrong_global_exp, X)

    # An exception should be raised when none of the input features is valid
    with pytest.raises(ValueError):
        append_composite_feature_importance(["Z", -1, 20], ebm, global_explanation, X)

    append_composite_feature_importance(composite_names, ebm, global_explanation, X)
    assert "A and B" in global_explanation._internal_obj["overall"]["names"]
    assert compute_composite_feature_importance(composite_names, ebm, X) in global_explanation._internal_obj["overall"]["scores"]

    append_composite_feature_importance(composite_names, ebm, global_explanation, X, composite_name="Comp 1")
    assert "Comp 1" in global_explanation._internal_obj["overall"]["names"]
    assert compute_composite_feature_importance(composite_names, ebm, X) in global_explanation._internal_obj["overall"]["scores"]

def _check_composite_feature_importance(X, y, ebm):
    composite_names = ["A", "B"]
    composite_indices = [0, 1]

    # An exception should be raised when the EBM is not fitted
    with pytest.raises(NotFittedError):
        compute_composite_feature_importance(composite_names, ebm, X)

    ebm.fit(X, y)

    # An exception should be raised when none of the input features is valid
    with pytest.raises(ValueError):
        compute_composite_feature_importance(["Z", -1, 10], ebm, X)

    with pytest.raises(ValueError):
        compute_composite_feature_importance([], ebm, X)

    # It should be the same for feature names and indices
    assert compute_composite_feature_importance(composite_names, ebm, X) == \
        compute_composite_feature_importance(composite_indices, ebm, X)

    # For one feature, its importance should be approx. equal to the one computed by interpret
    # TODO For multiclass this is currently consistent with interpret, but might be changed
    assert pytest.approx(ebm.get_importances()[0]) == compute_composite_feature_importance(["A"], ebm, X)

    mixed_list = ["A", "B", -2, 10]
    assert compute_composite_feature_importance(["A", "B"], ebm, X) == \
        compute_composite_feature_importance(mixed_list, ebm, X)

    _, contributions = ebm.predict_and_contrib(X)
    assert compute_composite_feature_importance(composite_names, ebm, X, contributions) == \
        compute_composite_feature_importance(composite_names, ebm, X)

def test_composite_feature_regression():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]

    ebm = ExplainableBoostingRegressor()
    _check_composite_feature_importance(X, y, ebm)

def test_composite_feature_classification():
    data = synthetic_classification()
    X = data["full"]["X"]
    y = data["full"]["y"]

    ebm = ExplainableBoostingClassifier()
    _check_composite_feature_importance(X, y, ebm)

def test_composite_feature_multiclass():
    data = synthetic_multiclass()
    X = data["full"]["X"]
    y = data["full"]["y"]

    ebm = ExplainableBoostingClassifier()
    _check_composite_feature_importance(X, y, ebm)