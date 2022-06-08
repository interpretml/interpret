import pandas as pd
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from ...ebm import ExplainableBoostingRegressor, ExplainableBoostingClassifier
from .....test.utils import synthetic_multiclass, synthetic_classification, synthetic_regression
from ..composite_importance import (
    compute_composite_importance,
    _get_composite_name,
    append_composite_importance,
    get_composite_and_individual_terms
)

def test_composite_name():
    mocked_ebm_term_names = ["Ft1", "Ft2", "Ft3", "Ft4", "Ft1 & Ft2"]

    assert "Ft3" == _get_composite_name(["Ft3"], mocked_ebm_term_names)

    composite_names = ["Ft1", "Ft3", "Ft1 & Ft2"]
    assert "Ft1, Ft3, Ft1 & Ft2" == _get_composite_name(composite_names, mocked_ebm_term_names)

    # Ft2, Ft4, Ft1 & Ft2
    composite_indices = [1, 3, 4]
    assert "Ft2, Ft4, Ft1 & Ft2" == _get_composite_name(composite_indices, mocked_ebm_term_names)

    out_of_bound_indices = [-1, 5]
    with pytest.raises(ValueError):
        _get_composite_name(out_of_bound_indices, mocked_ebm_term_names)

def test_append_composite_importance():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]
    composite_names = ["A", "B"]

    ebm = ExplainableBoostingRegressor()
    ebm.fit(X, y)

    # An exception should be raised when the EBM is not fitted
    non_fitted_ebm = ExplainableBoostingRegressor()
    with pytest.raises(NotFittedError):
        append_composite_importance(composite_names, non_fitted_ebm, X)

    # An exception should be raised when the explanation is not local
    local_exp = ebm.explain_local(X[:2])
    with pytest.raises(ValueError):
        append_composite_importance(composite_names, ebm, X, global_exp=local_exp)

    # An exception should be raised when none of the input terms is valid
    with pytest.raises(ValueError):
        append_composite_importance(["Z", -1, 20], ebm, X)

    global_explanation = append_composite_importance(composite_names, ebm, X)
    assert "A, B" in global_explanation._internal_obj["overall"]["names"]
    assert compute_composite_importance(composite_names, ebm, X) in global_explanation._internal_obj["overall"]["scores"]

    global_explanation = append_composite_importance(composite_names, ebm, X, composite_name="Comp 1")
    assert "Comp 1" in global_explanation._internal_obj["overall"]["names"]
    assert compute_composite_importance(composite_names, ebm, X) in global_explanation._internal_obj["overall"]["scores"]

def test_append_multiple_composite_importances():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]
    composite_terms_1 = ["A", "B"]
    composite_terms_2 = ["C", "D"]

    ebm = ExplainableBoostingRegressor()
    ebm.fit(X, y)

    global_explanation = append_composite_importance(composite_terms_1, ebm, X)
    global_explanation = append_composite_importance(composite_terms_2, ebm, X, global_exp=global_explanation)
    assert "A, B" in global_explanation._internal_obj["overall"]["names"]
    assert "C, D" in global_explanation._internal_obj["overall"]["names"]
    assert compute_composite_importance(composite_terms_1, ebm, X) in global_explanation._internal_obj["overall"]["scores"]
    assert compute_composite_importance(composite_terms_2, ebm, X) in global_explanation._internal_obj["overall"]["scores"]

def test_composite_and_individual_terms():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]
    composite_terms_1 = ["A", "B"]
    composite_terms_2 = ["C", "D"]

    ebm = ExplainableBoostingRegressor()
    ebm.fit(X, y)

    dict = get_composite_and_individual_terms(composite_terms_1, ebm, X)
    assert dict["A"] == compute_composite_importance(["A"], ebm, X)
    assert dict["B"] == compute_composite_importance(["B"], ebm, X)
    assert dict["A, B"] == compute_composite_importance(composite_terms_1, ebm, X)

    dict = get_composite_and_individual_terms([composite_terms_1], ebm, X)
    assert dict["A"] == compute_composite_importance(["A"], ebm, X)
    assert dict["B"] == compute_composite_importance(["B"], ebm, X)
    assert dict["A, B"] == compute_composite_importance(composite_terms_1, ebm, X)

    dict = get_composite_and_individual_terms([composite_terms_1, composite_terms_2], ebm, X)
    assert dict["A, B"] == compute_composite_importance(composite_terms_1, ebm, X)
    assert dict["C, D"] == compute_composite_importance(composite_terms_2, ebm, X)

def _check_composite_importance(X, y, ebm):
    composite_names = ["A", "B"]
    composite_indices = [0, 1]

    # An exception should be raised when the EBM is not fitted
    with pytest.raises(NotFittedError):
        compute_composite_importance(composite_names, ebm, X)

    ebm.fit(X, y)

    # An exception should be raised when at least one of the input terms is invalid
    with pytest.raises(ValueError):
        compute_composite_importance(["A", "B", 10], ebm, X)

    with pytest.raises(ValueError):
        compute_composite_importance([], ebm, X)

    # It should be the same for term names and indices
    assert compute_composite_importance(composite_names, ebm, X) == \
        compute_composite_importance(composite_indices, ebm, X)

    # For one term, its importance should be approx. equal to the one computed by interpret
    # TODO For multiclass this is currently consistent with interpret, but might be changed
    assert pytest.approx(ebm.get_importances()[0]) == compute_composite_importance(["A"], ebm, X)

    mixed_list = ["A", 1]
    assert compute_composite_importance(["A", "B"], ebm, X) == \
        compute_composite_importance(mixed_list, ebm, X)

    _, contributions = ebm.predict_and_contrib(X)
    assert compute_composite_importance(composite_names, ebm, X, contributions) == \
        compute_composite_importance(composite_names, ebm, X)

def test_composite_importance_regression():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]

    ebm = ExplainableBoostingRegressor()
    _check_composite_importance(X, y, ebm)

def test_composite_importance_classification():
    data = synthetic_classification()
    X = data["full"]["X"]
    y = data["full"]["y"]

    ebm = ExplainableBoostingClassifier()
    _check_composite_importance(X, y, ebm)

def test_composite_importance_multiclass():
    data = synthetic_multiclass()
    X = data["full"]["X"]
    y = data["full"]["y"]

    ebm = ExplainableBoostingClassifier()
    _check_composite_importance(X, y, ebm)
