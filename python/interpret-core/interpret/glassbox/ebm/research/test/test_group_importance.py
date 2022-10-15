import pandas as pd
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from ...ebm import ExplainableBoostingRegressor, ExplainableBoostingClassifier
from .....test.utils import synthetic_multiclass, synthetic_classification, synthetic_regression
from ..group_importance import (
    compute_group_importance,
    _get_group_name,
    append_group_importance,
    get_group_and_individual_importances,
    get_individual_importances,
    get_importance_per_top_groups
)

def test_group_name():
    mocked_ebm_term_list = ["Ft1", "Ft2", "Ft3", "Ft4", "Ft1 & Ft2"]

    assert "Ft3" == _get_group_name(["Ft3"], mocked_ebm_term_list)

    term_names = ["Ft1", "Ft3", "Ft1 & Ft2"]
    assert "Ft1, Ft3, Ft1 & Ft2" == _get_group_name(term_names, mocked_ebm_term_list)

    # Ft2, Ft4, Ft1 & Ft2
    term_indices = [1, 3, 4]
    assert "Ft2, Ft4, Ft1 & Ft2" == _get_group_name(term_indices, mocked_ebm_term_list)

    out_of_bound_indices = [-1, 5]
    with pytest.raises(ValueError):
        _get_group_name(out_of_bound_indices, mocked_ebm_term_list)

def test_append_group_importance():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]
    term_names = ["A", "B"]

    ebm = ExplainableBoostingRegressor()
    ebm.fit(X, y)

    # An exception should be raised when the EBM is not fitted
    non_fitted_ebm = ExplainableBoostingRegressor()
    with pytest.raises(NotFittedError):
        append_group_importance(term_names, non_fitted_ebm, X)

    # An exception should be raised when the explanation is not local
    local_exp = ebm.explain_local(X[:2])
    with pytest.raises(ValueError):
        append_group_importance(term_names, ebm, X, global_exp=local_exp)

    # An exception should be raised when none of the input terms is valid
    with pytest.raises(ValueError):
        append_group_importance(["Z", -1, 20], ebm, X)

    global_explanation = append_group_importance(term_names, ebm, X)
    assert "A, B" in global_explanation._internal_obj["overall"]["names"]
    assert compute_group_importance(term_names, ebm, X) in global_explanation._internal_obj["overall"]["scores"]

    global_explanation = append_group_importance(term_names, ebm, X, group_name="Comp 1")
    assert "Comp 1" in global_explanation._internal_obj["overall"]["names"]
    assert compute_group_importance(term_names, ebm, X) in global_explanation._internal_obj["overall"]["scores"]

def test_append_multiple_group_importances():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]
    term_names_1 = ["A", "B"]
    term_names_2 = ["C", "D"]

    ebm = ExplainableBoostingRegressor()
    ebm.fit(X, y)

    global_explanation = append_group_importance(term_names_1, ebm, X)
    global_explanation = append_group_importance(term_names_2, ebm, X, global_exp=global_explanation)
    assert "A, B" in global_explanation._internal_obj["overall"]["names"]
    assert "C, D" in global_explanation._internal_obj["overall"]["names"]
    assert compute_group_importance(term_names_1, ebm, X) in global_explanation._internal_obj["overall"]["scores"]
    assert compute_group_importance(term_names_2, ebm, X) in global_explanation._internal_obj["overall"]["scores"]

def test_append_same_importance_twice():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]
    term_names_1 = ["A", "B"]

    ebm = ExplainableBoostingRegressor()
    ebm.fit(X, y)

    global_explanation = append_group_importance(term_names_1, ebm, X)
    with pytest.raises(ValueError):
        global_explanation = append_group_importance(term_names_1, ebm, X, global_exp=global_explanation)

def test_group_and_individual_importances():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]
    term_names_1 = ["A", "B"]
    term_names_2 = ["C", "D"]

    ebm = ExplainableBoostingRegressor()
    ebm.fit(X, y)

    dict = get_group_and_individual_importances(term_names_1, ebm, X)
    assert dict["A"] == compute_group_importance(["A"], ebm, X)
    assert dict["B"] == compute_group_importance(["B"], ebm, X)
    assert dict["A, B"] == compute_group_importance(term_names_1, ebm, X)

    dict = get_group_and_individual_importances([term_names_1], ebm, X)
    assert dict["A"] == compute_group_importance(["A"], ebm, X)
    assert dict["B"] == compute_group_importance(["B"], ebm, X)
    assert dict["A, B"] == compute_group_importance(term_names_1, ebm, X)

    dict = get_group_and_individual_importances([term_names_1, term_names_2], ebm, X)
    assert dict["A, B"] == compute_group_importance(term_names_1, ebm, X)
    assert dict["C, D"] == compute_group_importance(term_names_2, ebm, X)

def test_individual_importances():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]

    ebm = ExplainableBoostingRegressor()
    ebm.fit(X, y)
    _, contributions = ebm.predict_and_contrib(X)

    dict = get_individual_importances(ebm, X, contributions)
    assert dict["A"] == compute_group_importance(["A"], ebm, X, contributions)
    assert dict["B"] == compute_group_importance(["B"], ebm, X, contributions)
    assert dict["C"] == compute_group_importance(["C"], ebm, X, contributions)
    assert dict["D"] == compute_group_importance(["D"], ebm, X, contributions)

def test_get_importance_per_top_groups():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]

    ebm = ExplainableBoostingRegressor()
    ebm.fit(X, y)

    df = get_importance_per_top_groups(ebm, X)
    dict = get_individual_importances(ebm, X)

    assert df.shape[0] == len(ebm.term_features_)
    # First group
    assert list(dict.keys())[0] in df["terms_per_group"][0]
    # Second group
    assert list(dict.keys())[0] in df["terms_per_group"][1]
    assert list(dict.keys())[1] in df["terms_per_group"][1]

def _check_group_importance(X, y, ebm):
    term_names = ["A", "B"]
    term_indices = [0, 1]

    # An exception should be raised when the EBM is not fitted
    with pytest.raises(NotFittedError):
        compute_group_importance(term_names, ebm, X)

    ebm.fit(X, y)

    # An exception should be raised when at least one of the input terms is invalid
    with pytest.raises(ValueError):
        compute_group_importance(["A", "B", 10], ebm, X)

    with pytest.raises(ValueError):
        compute_group_importance([], ebm, X)

    # It should be the same for term names and indices
    assert compute_group_importance(term_names, ebm, X) == \
        compute_group_importance(term_indices, ebm, X)

    # For one term, its importance should be approx. equal to the one computed by interpret
    # TODO For multiclass this is currently consistent with interpret, but might be changed
    assert pytest.approx(ebm.term_importances()[0]) == compute_group_importance(["A"], ebm, X)

    mixed_list = ["A", 1]
    assert compute_group_importance(["A", "B"], ebm, X) == \
        compute_group_importance(mixed_list, ebm, X)

    _, contributions = ebm.predict_and_contrib(X)
    assert compute_group_importance(term_names, ebm, X, contributions) == \
        compute_group_importance(term_names, ebm, X)

def test_group_importance_regression():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]

    ebm = ExplainableBoostingRegressor()
    _check_group_importance(X, y, ebm)

def test_group_importance_classification():
    data = synthetic_classification()
    X = data["full"]["X"]
    y = data["full"]["y"]

    ebm = ExplainableBoostingClassifier()
    _check_group_importance(X, y, ebm)

def test_group_importance_multiclass():
    data = synthetic_multiclass()
    X = data["full"]["X"]
    y = data["full"]["y"]

    ebm = ExplainableBoostingClassifier()
    _check_group_importance(X, y, ebm)
