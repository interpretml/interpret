# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

# TODO PK add a test for Regression with interactions
# TODO PK add a test with a real regression dataset
# TODO PK add a test with more than 1 multiclass interaction

from ....test.utils import (
    synthetic_multiclass,
    synthetic_classification,
    adult_classification,
    iris_classification,
)
from ....test.utils import synthetic_regression
from ..ebm import ExplainableBoostingRegressor, ExplainableBoostingClassifier, DPExplainableBoostingClassifier, DPExplainableBoostingRegressor

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    cross_validate,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.metrics import accuracy_score
from sklearn.utils.estimator_checks import check_estimator
import pytest

import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn

def valid_ebm(ebm):
    assert ebm.term_features_[0] == (0,)

    for term_scores in ebm.term_scores_:
        all_finite = np.isfinite(term_scores).all()
        assert all_finite

def _smoke_test_explanations(global_exp, local_exp, port):
    from .... import preserve, show, shutdown_show_server, set_show_addr

    set_show_addr(("127.0.0.1", port))

    # Smoke test: should run without crashing.
    preserve(global_exp)
    preserve(local_exp)
    show(global_exp)
    show(local_exp)

    # Check all features for global (including interactions).
    for selector_key in global_exp.selector[global_exp.selector.columns[0]]:
        preserve(global_exp, selector_key)

    shutdown_show_server()


@pytest.mark.slow
def test_unknown_multiclass_category():
    data = iris_classification()
    X_train = data["train"]["X"]
    y_train = data["train"]["y"]

    X_test = data["test"]["X"]
    y_test = data["test"]["y"]

    # Add categorical feature
    X_train['cat_feature'] = [np.random.choice(['a', 'b', 'c']) for x in range(X_train.shape[0])]
    X_test['cat_feature'] = ['d' for x in range(X_test.shape[0])]  # Unknown category in test set

    # X_train['cat_feature'][1] = np.nan
    # X_test['cat_feature'][1] = np.nan

    clf = ExplainableBoostingClassifier()
    clf.fit(X_train, y_train)

    # Term contributions for categorical feature should always be 0 in test
    assert np.all(clf.explain_local(X_train).data(0)['scores'][-1] != 0)
    assert np.all(clf.explain_local(X_test).data(0)['scores'][-1] == 0)

@pytest.mark.slow
def test_unknown_binary_category():
    data = adult_classification()
    X_tr = data["train"]["X"]
    y_tr = data["train"]["y"]
    X_te = data["test"]["X"]
    y_te = data["test"]["y"]

    ebm = ExplainableBoostingClassifier(n_jobs=2, outer_bags=2, interactions=[[0, 13], [1, 2], [13, 3]])
    ebm.fit(X_tr, y_tr)

    test_point = X_te[[0]].copy()
    perturbed_point = test_point.copy()
    perturbed_point[0, -1] = 'Unseen Categorical'  # Change country to unseen value

    # Perturbed feature contribution
    country_contrib = ebm.explain_local(test_point).data(0)['scores'][-4]
    perturbed_contrib = ebm.explain_local(perturbed_point).data(0)['scores'][-4]

    assert country_contrib != 0
    assert perturbed_contrib == 0

    # Perturbed interaction contribution (dim 1)
    country_inter_contrib = ebm.explain_local(test_point).data(0)['scores'][-3]
    perturbed_inter_contrib = ebm.explain_local(perturbed_point).data(0)['scores'][-3]

    assert country_inter_contrib != 0
    assert perturbed_inter_contrib == 0

    # Perturbed interaction contribution (dim 2)
    country_inter_contrib_2 = ebm.explain_local(test_point).data(0)['scores'][-1]
    perturbed_inter_contrib_2 = ebm.explain_local(perturbed_point).data(0)['scores'][-1]

    assert country_inter_contrib_2 != 0
    assert perturbed_inter_contrib_2 == 0

    # Sum(logit) differences from decision_function should only come from perturbed columns
    test_logit = ebm.decision_function(test_point)
    perturbed_logit = ebm.decision_function(perturbed_point)

    assert test_logit != perturbed_logit
    assert np.allclose(test_logit, (perturbed_logit + country_contrib + country_inter_contrib + country_inter_contrib_2))


@pytest.mark.visual
@pytest.mark.slow
def test_ebm_synthetic_multiclass():
    data = synthetic_multiclass()
    X_train = data["train"]["X"]
    y_train = data["train"]["y"]
    X_test = data["test"]["X"]
    y_test = data["test"]["y"]

    clf = ExplainableBoostingClassifier(n_jobs=-2, interactions=0, outer_bags=2)
    clf.fit(X_train, y_train)

    prob_scores = clf.predict_proba(X_train)

    within_bounds = (prob_scores >= 0.0).all() and (prob_scores <= 1.0).all()
    assert within_bounds

    valid_ebm(clf)

    # Smoke test visualization(s)
    ebm_global = clf.explain_global()
    ebm_global.visualize(None)
    fig = ebm_global.visualize(0)
    assert len(fig.data) == 4  # Number of features

    ebm_local = clf.explain_local(X_test, y_test)
    ebm_local.visualize(None)
    fig = ebm_local.visualize(0)
    assert len(fig.data) == 3  # Number of classes


@pytest.mark.slow
def test_ebm_synthetic_multiclass_pairwise():
    data = synthetic_multiclass()
    X = data["full"]["X"]
    y = data["full"]["y"]

    clf = ExplainableBoostingClassifier(n_jobs=-2, interactions=1, outer_bags=2)
    clf.fit(X, y)
    clf.predict_proba(X)
    valid_ebm(clf)


@pytest.mark.slow
def test_ebm_synthetic_pairwise():
    a = np.random.randint(low=0, high=50, size=1000)
    b = np.random.randint(low=0, high=20, size=1000)

    df = pd.DataFrame(np.c_[a, b], columns=["a", "b"])
    df["y"] = [
        1 if (x > 35 and y > 15) or (x < 15 and y < 5) else 0
        for x, y in zip(df["a"], df["b"])
    ]

    X = df[["a", "b"]]
    y = df["y"]

    random_state = 1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state
    )

    clf = ExplainableBoostingClassifier(n_jobs=1, outer_bags=1, interactions=1)
    clf.fit(X_train, y_train)

    clf_global = clf.explain_global()

    # Low/Low and High/High should learn high scores
    assert clf_global.data(2)["scores"][-1][-1] > 5
    assert clf_global.data(2)["scores"][1][1] > 5


def test_ebm_tripple():
    data = iris_classification()
    X_train = data["train"]["X"]
    y_train = data["train"]["y"]

    X_test = data["test"]["X"]
    y_test = data["test"]["y"]

    # iris is multiclass, but for now pretend this is a regression problem
    clf = ExplainableBoostingRegressor(interactions=[(0,1,2), (0,1,3), (1,2,3), (0, 1)])
    clf.fit(X_train, y_train)
    scores = clf.predict(X_test)
    valid_ebm(clf)

@pytest.mark.slow
def test_prefit_ebm():
    data = synthetic_classification()
    X = data["full"]["X"]
    y = data["full"]["y"]

    clf = ExplainableBoostingClassifier(n_jobs=1, interactions=0, max_rounds=0)
    clf.fit(X, y)

    for term_scores in clf.term_scores_:
        has_non_zero = np.any(term_scores)
        assert not has_non_zero


def test_ebm_synthetic_regression():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]

    clf = ExplainableBoostingRegressor(n_jobs=-2, interactions=0)
    clf.fit(X, y)
    clf.predict(X)

    valid_ebm(clf)

def test_ebm_synthetic_classification():
    data = synthetic_classification()
    X = data["full"]["X"]
    y = data["full"]["y"]

    clf = ExplainableBoostingClassifier(n_jobs=-2, interactions=0)
    clf.fit(X, y)
    prob_scores = clf.predict_proba(X)

    within_bounds = (prob_scores >= 0.0).all() and (prob_scores <= 1.0).all()
    assert within_bounds

    valid_ebm(clf)

def test_ebm_missing():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]

    X[0, 0] = np.nan

    clf = ExplainableBoostingRegressor(n_jobs=-2, interactions=0)
    clf.fit(X, y)
    clf.predict(X)

    valid_ebm(clf)

def test_ebm_only_missing():
    X = np.full((10,10), np.nan)
    y = np.full(10, 0)
    y[0] = 1

    clf = ExplainableBoostingClassifier(n_jobs=1)
    clf.fit(X, y)
    clf.predict(X)
    global_exp = clf.explain_global()
    local_exp = clf.explain_local(X, y)


def test_ebm_synthetic_singleclass_classification():
    data = synthetic_classification()
    X = data["full"]["X"]
    y = np.zeros(X.shape[0], np.bool_)

    clf = ExplainableBoostingClassifier(n_jobs=-2, interactions=0)
    clf.fit(X, y)

    prob_scores = clf.predict_proba(X)
    assert prob_scores.ndim == 2
    assert prob_scores.shape[0] == len(y)
    assert prob_scores.shape[1] == 1
    assert (prob_scores == 1.0).all()

    predicts = clf.predict(X)
    assert predicts.ndim == 1
    assert predicts.shape[0] == len(y)
    assert (predicts == False).all()

    scores = clf.decision_function(X)
    assert scores.ndim == 1
    assert scores.shape[0] == len(y)
    assert (scores == -np.inf).all()

    prob_scores, explanations = clf.predict_and_contrib(X, output='probabilities')
    assert prob_scores.ndim == 2
    assert prob_scores.shape[0] == len(y)
    assert prob_scores.shape[1] == 1
    assert (prob_scores == 1.0).all()

    scores, explanations = clf.predict_and_contrib(X, output='logits')
    assert scores.ndim == 1
    assert scores.shape[0] == len(y)
    assert (scores == -np.inf).all()

    predicts, explanations = clf.predict_and_contrib(X, output='labels')
    assert predicts.ndim == 1
    assert predicts.shape[0] == len(y)
    assert (predicts == False).all()

@pytest.mark.visual
@pytest.mark.slow
def test_ebm_uniform():
    from sklearn.metrics import roc_auc_score

    data = adult_classification()
    X = data["full"]["X"]
    y = data["full"]["y"]
    X_tr = data["train"]["X"]
    y_tr = data["train"]["y"]
    X_te = data["test"]["X"]
    y_te = data["test"]["y"]

    clf = ExplainableBoostingClassifier(binning="uniform", n_jobs=-2, interactions=3)
    n_splits = 3
    ss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.25, random_state=1337)
    cross_validate(
        clf, X, y, scoring="roc_auc", cv=ss, n_jobs=None, return_estimator=True
    )

    clf = ExplainableBoostingClassifier(binning="uniform", n_jobs=-2, interactions=3)
    clf.fit(X_tr, y_tr)

    prob_scores = clf.predict_proba(X_te)

    within_bounds = (prob_scores >= 0.0).all() and (prob_scores <= 1.0).all()
    assert within_bounds

    # Performance
    auc = roc_auc_score(y_te, prob_scores[:, 1])
    assert auc > 0.5

    valid_ebm(clf)

    global_exp = clf.explain_global()
    local_exp = clf.explain_local(X_te[:5, :], y_te[:5])

    _smoke_test_explanations(global_exp, local_exp, 6000)

@pytest.mark.visual
@pytest.mark.slow
def test_ebm_uniform_multiclass():
    data = iris_classification()
    X_train = data["train"]["X"]
    y_train = data["train"]["y"]

    X_test = data["test"]["X"]
    y_test = data["test"]["y"]

    clf = ExplainableBoostingClassifier(binning="uniform")
    clf.fit(X_train, y_train)

    assert accuracy_score(y_test, clf.predict(X_test)) > 0.9

    global_exp = clf.explain_global()
    local_exp = clf.explain_local(X_test, y_test)

    _smoke_test_explanations(global_exp, local_exp, 6001)


@pytest.mark.visual
@pytest.mark.slow
def test_ebm_adult():
    from sklearn.metrics import roc_auc_score

    data = adult_classification()
    X = data["full"]["X"]
    y = data["full"]["y"]
    X_tr = data["train"]["X"]
    y_tr = data["train"]["y"]
    X_te = data["test"]["X"]
    y_te = data["test"]["y"]

    clf = ExplainableBoostingClassifier(n_jobs=-2, interactions=3)
    n_splits = 3
    ss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.25, random_state=1337)
    cross_validate(
        clf, X, y, scoring="roc_auc", cv=ss, n_jobs=None, return_estimator=True
    )

    clf = ExplainableBoostingClassifier(n_jobs=-2, interactions=3)
    clf.fit(X_tr, y_tr)

    prob_scores = clf.predict_proba(X_te)

    within_bounds = (prob_scores >= 0.0).all() and (prob_scores <= 1.0).all()
    assert within_bounds

    # Performance
    auc = roc_auc_score(y_te, prob_scores[:, 1])
    assert auc > 0.5

    valid_ebm(clf)

    global_exp = clf.explain_global()
    local_exp = clf.explain_local(X_te[:5, :], y_te[:5])

    _smoke_test_explanations(global_exp, local_exp, 6000)


def test_ebm_predict_and_contrib_proba():
    data = adult_classification()
    X_tr = data["train"]["X"]
    y_tr = data["train"]["y"]
    X_te = data["test"]["X"]

    clf = ExplainableBoostingClassifier(n_jobs=-2, interactions=3)
    clf.fit(X_tr, y_tr)

    probabilities_orig = clf.predict_proba(X_te)
    probabilities, explanations = clf.predict_and_contrib(X_te, output='probabilities')

    assert np.allclose(probabilities_orig, probabilities)

    # TODO: Make a better test to ensure explanations are correct
    explanations_sum_orig = clf.decision_function(X_te)
    explanations_sum = np.sum(explanations, axis=1)
    explanations_sum += clf.intercept_

    assert np.allclose(explanations_sum_orig, explanations_sum)


def test_ebm_predict_and_contrib_logits():
    data = adult_classification()
    X_tr = data["train"]["X"]
    y_tr = data["train"]["y"]
    X_te = data["test"]["X"]

    clf = ExplainableBoostingClassifier(n_jobs=-2, interactions=3)
    clf.fit(X_tr, y_tr)

    logits_orig = clf.decision_function(X_te)
    logits, explanations = clf.predict_and_contrib(X_te, output='logits')

    assert np.allclose(logits_orig, logits)

    # TODO: Make a better test to ensure explanations are correct
    explanations_sum = np.sum(explanations, axis=1)
    explanations_sum += clf.intercept_

    assert np.allclose(logits_orig, explanations_sum)

def test_ebm_predict_and_contrib_labels():
    data = adult_classification()
    X_tr = data["train"]["X"]
    y_tr = data["train"]["y"]
    X_te = data["test"]["X"]

    clf = ExplainableBoostingClassifier(n_jobs=-2, interactions=3)
    clf.fit(X_tr, y_tr)

    labels_orig = clf.predict(X_te)
    labels, explanations = clf.predict_and_contrib(X_te, 'labels')

    assert np.array_equal(labels_orig, labels)

    # TODO: Make a better test to ensure explanations are correct
    explanations_sum_orig = clf.decision_function(X_te)
    explanations_sum = np.sum(explanations, axis=1)
    explanations_sum += clf.intercept_

    assert np.allclose(explanations_sum_orig, explanations_sum)


def test_ebm_predict_and_contrib_regression():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]

    clf = ExplainableBoostingRegressor(n_jobs=-2, interactions=0)
    clf.fit(X, y)

    predictions_orig = clf.predict(X)
    predictions, explanations = clf.predict_and_contrib(X)

    assert np.allclose(predictions_orig, predictions)

    explanations_sum = np.sum(explanations, axis=1)
    explanations_sum += clf.intercept_
    assert np.allclose(predictions_orig, explanations_sum)


def test_ebm_sample_weight():
    data = adult_classification()
    X_train = data["train"]["X"][:, [0, 1]]
    y_train = data["train"]["y"]

    X_test = data["test"]["X"][:, [0, 1]]
    y_test = data["test"]["y"]

    w_train = np.ones_like(y_train)
    w_train[0] = 10

    # Smoke test defaults (bagging, parallelization, interactions)
    clf_default = ExplainableBoostingClassifier()
    clf_default.fit(X_train, y_train, sample_weight=w_train)

    # Minimal EBM to verify exact sample weight behavior
    clf = ExplainableBoostingClassifier(outer_bags=1, validation_size=0, early_stopping_rounds=-1, max_rounds=100, n_jobs=1)
    clf.fit(X_train, y_train, sample_weight=w_train)

    # Create 10 manual copies of X_train[0]
    repeat_indexes = [0] * 9 + list(range(len(X_train)))
    X_train_u = X_train[repeat_indexes]
    y_train_u = y_train.iloc[repeat_indexes]

    clf_u = ExplainableBoostingClassifier(outer_bags=1, validation_size=0, early_stopping_rounds=-1, max_rounds=100, n_jobs=1)
    clf_u.fit(X_train_u, y_train_u)

    assert np.allclose(clf.predict_proba(X_test), clf_u.predict_proba(X_test))

@pytest.mark.visual
@pytest.mark.slow
def test_ebm_iris():
    data = iris_classification()
    X_train = data["train"]["X"]
    y_train = data["train"]["y"]

    X_test = data["test"]["X"]
    y_test = data["test"]["y"]

    clf = ExplainableBoostingClassifier()
    clf.fit(X_train, y_train)

    assert accuracy_score(y_test, clf.predict(X_test)) > 0.9

    global_exp = clf.explain_global()
    local_exp = clf.explain_local(X_test, y_test)

    _smoke_test_explanations(global_exp, local_exp, 6001)


@pytest.mark.visual
@pytest.mark.slow
def test_ebm_sparse():
    """ Validate running EBM on scipy sparse data
    """
    from sklearn.datasets import make_multilabel_classification

    np.random.seed(0)
    n_features = 5
    X, y = make_multilabel_classification(
        n_samples=20, sparse=True, n_features=n_features, n_classes=1, n_labels=2
    )

    # train linear model
    clf = ExplainableBoostingClassifier()
    clf.fit(X, y)

    assert accuracy_score(y, clf.predict(X)) >= 0.8
    global_exp = clf.explain_global()
    local_exp = clf.explain_local(X, y)
    _smoke_test_explanations(global_exp, local_exp, 6002)


@pytest.mark.slow
def test_zero_validation():
    data = synthetic_classification()
    X = data["full"]["X"]
    y = data["full"]["y"]

    clf = ExplainableBoostingClassifier(n_jobs=1, interactions=2, validation_size=0)
    clf.fit(X, y)

@pytest.mark.visual
@pytest.mark.slow
def test_dp_ebm_adult():
    from sklearn.metrics import roc_auc_score
    from ..ebm import DPExplainableBoostingClassifier

    data = adult_classification(sample=1)
    X = data["full"]["X"]
    y = data["full"]["y"]
    X_tr = data["train"]["X"]
    y_tr = data["train"]["y"]
    X_te = data["test"]["X"]
    y_te = data["test"]["y"]
    w_tr = np.ones_like(y_tr)
    w_tr[-1] = 2

    clf = DPExplainableBoostingClassifier(epsilon=1)
    n_splits = 3
    ss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.25, random_state=1337)
    res = cross_validate(
        clf, X, y, scoring="roc_auc", cv=ss, n_jobs=None, return_estimator=True
    )

    clf = DPExplainableBoostingClassifier(epsilon=1)
    clf.fit(X_tr, y_tr, w_tr)

    prob_scores = clf.predict_proba(X_te)

    within_bounds = (prob_scores >= 0.0).all() and (prob_scores <= 1.0).all()
    assert within_bounds

    # Performance
    auc = roc_auc_score(y_te, prob_scores[:, 1])
    assert auc > 0.5

    valid_ebm(clf)

    global_exp = clf.explain_global()
    local_exp = clf.explain_local(X_te[:5, :], y_te[:5])

    _smoke_test_explanations(global_exp, local_exp, 6000)

def test_dp_ebm_synthetic_regression():
    from ..ebm import DPExplainableBoostingRegressor

    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]
    w = np.ones_like(y)
    w[-1] = 2

    clf = DPExplainableBoostingRegressor()
    clf.fit(X, y, w)
    clf.predict(X)

    valid_ebm(clf)

def test_dp_ebm_external_privacy_schema():
    from ..ebm import DPExplainableBoostingRegressor

    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]

    # synthetic regression is all sampled from N(0, 1)
    privacy_schema = {
        0: (-3, 3),
        1: (-3, 3),
        2: (-3, 3),
        3: (-3, 3),
        'target': (-3, 3)
    }

    clf = DPExplainableBoostingRegressor(privacy_schema=privacy_schema)
    clf.fit(X, y)
    clf.predict(X)

    valid_ebm(clf)

@pytest.mark.slow
def test_ebm_calibrated_classifier_cv():
    """ Tests if unsigned integers can be handled when
        using CalibratedClassifierCV.
    """
    from sklearn.calibration import CalibratedClassifierCV

    X = np.array([[0, 1, 0, 0],
                  [0, 0, 0, 1],
                  [1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1],
                  [1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [1, 0, 0, 0],
                  [0, 0, 0, 1]],
                  dtype=np.uint8)

    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                 dtype=np.uint8)

    clf = ExplainableBoostingClassifier()
    calib = CalibratedClassifierCV(clf)
    calib.fit(X, y)

def test_ebm_unknown_value_at_predict():
    """ Tests if unsigned integers can be handled when unknown values
        are found by predict.

        e.g. feature 3 has only 0's in X but a 1 in X_test.
    """
    X = np.array([[0, 1, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0],
                  [1, 0, 0, 0],
                  [0, 0, 0, 1]],
                  dtype=np.uint8)

    X_test = np.array([[0, 1, 0, 0],
                       [0, 0, 1, 1],
                       [1, 0, 0, 0]],
                       dtype=np.uint8)

    y = np.array([0, 1, 1, 1, 1],
                 dtype=np.uint8)

    clf = ExplainableBoostingClassifier()
    clf.fit(X, y)
    clf.predict(X_test)

    valid_ebm(clf)

@pytest.mark.skip(reason="can't run this test reliably until we depend on scikit-learn 0.22")
def test_scikit_learn_compatibility():
    """ Run scikit-learn compatibility tests
    """

    # sklearn tests in:
    # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/estimator_checks.py    
    
    skip_tests = { 
        'check_dtype_object', # the error message required to pass is too specific and incorrect for us
        'check_classifiers_one_label', # TODO: fix this!  We should accept 1 category
        'check_classifiers_regression_target', # we're more permissive and convert any y values to str
        'check_supervised_y_no_nan', # error message too specific
        'check_supervised_y_2d', # we ignore useless added dimensions
        'check_fit2d_predict1d', # we accept 1d for predict
        'check_fit2d_1sample', # TODO: we allow fitting on 1 sample, but this kind of input is likely a bug from the caller, so change this
        'check_regressors_no_decision_function', # TODO: fix this!
    }
    for estimator, check_func in check_estimator(ExplainableBoostingClassifier(), generate_only=True): 
        f = check_func.func 
        module = f.__module__
        shortname = f.__name__
        fullname = f"{module}.{shortname}"
        if shortname not in skip_tests:
            try:
                check_func(estimator) 
            except BaseException as e:
                print(fullname)
                print(f"{type(e).__name__}: {e}") 
                print() 

    for estimator, check_func in check_estimator(ExplainableBoostingRegressor(), generate_only=True): 
        f = check_func.func 
        module = f.__module__
        shortname = f.__name__
        fullname = f"{module}.{shortname}"
        if shortname not in skip_tests:
            try:
                check_func(estimator) 
            except BaseException as e:
                print(fullname)
                print(f"{type(e).__name__}: {e}") 
                print() 

    for estimator, check_func in check_estimator(DPExplainableBoostingClassifier(), generate_only=True): 
        f = check_func.func 
        module = f.__module__
        shortname = f.__name__
        fullname = f"{module}.{shortname}"
        if shortname not in skip_tests:
            try:
                check_func(estimator) 
            except BaseException as e:
                print(fullname)
                print(f"{type(e).__name__}: {e}") 
                print() 

    for estimator, check_func in check_estimator(DPExplainableBoostingRegressor(), generate_only=True): 
        f = check_func.func 
        module = f.__module__
        shortname = f.__name__
        fullname = f"{module}.{shortname}"
        if shortname not in skip_tests:
            try:
                check_func(estimator) 
            except BaseException as e:
                print(fullname)
                print(f"{type(e).__name__}: {e}") 
                print() 

def test_json_classification():
    data = synthetic_classification()
    X = data["full"]["X"]
    y = data["full"]["y"]

    X["A"] = pd.cut(X["A"], [-np.inf, -1, -0.5, 0.5, 1, np.inf], labels=["apples", "oranges", "0", "almonds", "peanuts"], ordered=False)
    X["B"] = pd.cut(X["B"], [-np.inf, -0.5, 0.5, np.inf], labels=["low", "medium", "high"], ordered=True)
    X["C"] = X["C"].mask(X['C'] < 0, 0)

    clf = ExplainableBoostingClassifier(outer_bags=3, max_bins=10, max_interaction_bins=4, interactions=[(1,2), (2, 0)])
    clf.fit(X, y)

    # combine the last two bins into one bin
    max_val = max(clf.bins_[0][0].values())
    clf.bins_[0][0] = { key: value if value != max_val else max_val - 1 for key, value in clf.bins_[0][0].items() }

    clf.term_scores_[0] = np.delete(clf.term_scores_[0], 1)
    clf.term_scores_[0][1] = -np.inf
    clf.term_scores_[0][2] = np.inf
    clf.term_scores_[0][3] = np.nan

    clf.standard_deviations_[0] = np.delete(clf.standard_deviations_[0], 1)
    clf.bagged_scores_[0] = np.array([np.delete(clf.bagged_scores_[0][0], 1), np.delete(clf.bagged_scores_[0][1], 1), np.delete(clf.bagged_scores_[0][2], 1)])

    clf.bin_weights_[0] = np.delete(clf.bin_weights_[0], 1)

    json_text = clf._to_json(properties='all')

def test_json_multiclass():
    data = synthetic_multiclass()
    X = data["full"]["X"]
    y = data["full"]["y"]
    feature_types = ['continuous'] * X.shape[1]
    feature_types[0] = 'nominal'
    clf = ExplainableBoostingClassifier(max_bins=10, feature_types=feature_types, interactions=0)
    clf.fit(X, y)
    json_text = clf._to_json(properties='all')

def test_json_regression():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]
    feature_types = ['continuous'] * X.shape[1]
    feature_types[0] = 'nominal'
    clf = ExplainableBoostingRegressor(max_bins=5, max_interaction_bins=4, feature_types=feature_types, interactions=[(1,2), (2,3)])
    clf.fit(X, y)
    json_text = clf._to_json(properties='all')

def test_json_dp_classification():
    data = synthetic_classification()
    X = data["full"]["X"]
    y = data["full"]["y"]
    feature_types = ['continuous'] * X.shape[1]
    feature_types[0] = 'nominal'
    clf = DPExplainableBoostingClassifier(max_bins=10, feature_types=feature_types)
    clf.fit(X, y)
    clf.term_scores_[0][0] = np.nan
    clf.term_scores_[0][1] = np.inf
    clf.term_scores_[0][2] = -np.inf
    json_text = clf._to_json(properties='all')

def test_json_dp_regression():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]
    feature_types = ['continuous'] * X.shape[1]
    feature_types[0] = 'nominal'
    clf = DPExplainableBoostingRegressor(max_bins=5, feature_types=feature_types)
    clf.fit(X, y)
    json_text = clf._to_json(properties='all')
