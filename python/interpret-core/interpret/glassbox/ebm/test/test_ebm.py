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
from ..ebm import ExplainableBoostingRegressor, ExplainableBoostingClassifier

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    cross_validate,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.metrics import accuracy_score
import pytest

import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn


@pytest.mark.slow
def test_ebm_synthetic_multiclass():
    data = synthetic_multiclass()
    X = data["full"]["X"]
    y = data["full"]["y"]

    clf = ExplainableBoostingClassifier(n_jobs=-2, interactions=0, outer_bags=2)
    clf.fit(X, y)

    prob_scores = clf.predict_proba(X)

    within_bounds = (prob_scores >= 0.0).all() and (prob_scores <= 1.0).all()
    assert within_bounds

    valid_ebm(clf)


@pytest.mark.slow
def test_ebm_synthetic_multiclass_pairwise():
    data = synthetic_multiclass()
    X = data["full"]["X"]
    y = data["full"]["y"]

    clf = ExplainableBoostingClassifier(n_jobs=-2, interactions=1, outer_bags=2)
    with pytest.raises(RuntimeError):
        clf.fit(X, y)


@pytest.mark.slow
def test_ebm_synthetic_pairwise():
    a = np.random.randint(low=0, high=50, size=10000)
    b = np.random.randint(low=0, high=20, size=10000)

    df = pd.DataFrame(np.c_[a, b], columns=["a", "b"])
    df["y"] = [
        1 if (x > 35 and y > 15) or (x < 15 and y < 5) else 0
        for x, y in zip(df["a"], df["b"])
    ]

    X = df[["a", "b"]]
    y = df["y"]

    seed = 1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed
    )

    clf = ExplainableBoostingClassifier(interactions=1)
    clf.fit(X_train, y_train)

    clf_global = clf.explain_global()

    # Low/Low and High/High should learn high scores
    assert clf_global.data(2)["scores"][-1][-1] > 5
    assert clf_global.data(2)["scores"][0][0] > 5


@pytest.mark.slow
def test_prefit_ebm():
    data = synthetic_classification()
    X = data["full"]["X"]
    y = data["full"]["y"]

    clf = ExplainableBoostingClassifier(n_jobs=1, interactions=0, max_rounds=0)
    clf.fit(X, y)

    for _, model_feature_combination in enumerate(clf.additive_terms_):
        has_non_zero = np.any(model_feature_combination)
        assert not has_non_zero


def test_ebm_synthetic_regression():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]

    clf = ExplainableBoostingRegressor(n_jobs=-2, interactions=0)
    clf.fit(X, y)
    clf.predict(X)

    valid_ebm(clf)


def valid_ebm(ebm):
    assert ebm.feature_groups_[0] == [0]

    for _, model_feature_combination in enumerate(ebm.additive_terms_):
        all_finite = np.isfinite(model_feature_combination).all()
        assert all_finite


def test_ebm_synthetic_classfication():
    data = synthetic_classification()
    X = data["full"]["X"]
    y = data["full"]["y"]

    clf = ExplainableBoostingClassifier(n_jobs=-2, interactions=0)
    clf.fit(X, y)
    prob_scores = clf.predict_proba(X)

    within_bounds = (prob_scores >= 0.0).all() and (prob_scores <= 1.0).all()
    assert within_bounds

    valid_ebm(clf)


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

    assert accuracy_score(y, clf.predict(X)) > 0.9
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
