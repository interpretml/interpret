# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

# TODO PK add a test for Regression with interactions
# TODO PK add a test with a real regression dataset
# TODO PK add a test with more than 1 multiclass interaction

import math
from sklearn.metrics import log_loss

from interpret.utils import make_synthetic
from ...tutils import (
    synthetic_multiclass,
    synthetic_classification,
    iris_classification,
    smoke_test_explanations,
)
from ...tutils import synthetic_regression
from interpret.glassbox import (
    ExplainableBoostingRegressor,
    ExplainableBoostingClassifier,
)
from interpret.privacy import (
    DPExplainableBoostingClassifier,
    DPExplainableBoostingRegressor,
)
from interpret.utils import inv_link

from io import StringIO
import json
import numpy as np
import pandas as pd  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
from sklearn.utils.estimator_checks import check_estimator  # type: ignore
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


def test_binarize_1term():
    data = iris_classification()
    X_train = data["train"]["X"]
    y_train = data["train"]["y"]
    X_test = data["test"]["X"]
    y_test = data["test"]["y"]

    clf = ExplainableBoostingClassifier(interactions=0)
    clf.fit(X_train, y_train)

    # an EBM with only one term should remain identical if we use passthrough
    clf.remove_terms(range(1, len(clf.term_scores_)))

    # slightly unbalance the EBM so that it is not centered anymore through editing
    clf.term_scores_[0][1, 0] = 10

    original_pred = clf.predict_proba(X_test)
    clf._ovrize(1.0)
    assert np.allclose(original_pred, clf.predict_proba(X_test))
    clf._multinomialize(1.0)
    assert np.allclose(original_pred, clf.predict_proba(X_test))


def test_vlogit_2class():
    data = synthetic_classification()
    X_train = data["train"]["X"]
    y_train = data["train"]["y"]
    X_test = data["test"]["X"]
    y_test = data["test"]["y"]

    clf = ExplainableBoostingClassifier(interactions=10)
    clf.fit(X_train, y_train)

    # slightly unbalance the EBM so that it is not centered anymore through editing
    clf.term_scores_[0][1] = 10

    original_pred = clf.predict_proba(X_test)

    # hack the EBM into being a 2-class OVR, which is not legal, but will work
    mod = clf.copy()
    mod.standard_deviations_ = None
    mod.bagged_scores_ = None
    mod.bagged_intercept_ = None
    for i in range(len(mod.term_scores_)):
        term = np.expand_dims(mod.term_scores_[i], axis=-1)
        mod.term_scores_[i] = np.c_[-term, term]
    mod.intercept_ = np.array([-mod.intercept_[0], mod.intercept_[0]], np.float64)
    mod.link_ = "vlogit"

    assert np.allclose(original_pred, mod.predict_proba(X_test))
    assert np.allclose(original_pred, mod._binarize(0.75)[1].predict_proba(X_test))
    mod._multinomialize(0.625)
    assert np.allclose(original_pred, mod.predict_proba(X_test))
    assert np.allclose(original_pred, mod._binarize(0.5)[1].predict_proba(X_test))
    mod._ovrize(0.125)
    assert np.allclose(original_pred, mod.predict_proba(X_test))
    assert np.allclose(original_pred, mod._binarize(0.25)[1].predict_proba(X_test))


def test_binarize():
    data = iris_classification()
    X_train = data["train"]["X"]
    y_train = data["train"]["y"]
    X_test = data["test"]["X"]
    y_test = data["test"]["y"]

    clf = ExplainableBoostingClassifier(interactions=0)
    clf.fit(X_train, y_train)

    # slightly unbalance the EBM so that it is not centered anymore through editing
    clf.term_scores_[0][1, 0] = 10

    logloss_multinomial = log_loss(y_test, clf.predict_proba(X_test))

    ovr = clf.copy()._ovrize()
    ebms = clf.copy()._binarize()

    probas = np.array([ebm.predict_proba(X_test)[:, 1] for ebm in ebms]).T
    probas /= np.sum(probas, axis=-1, keepdims=True)

    logloss_binary = log_loss(y_test, probas)
    ratio = logloss_binary / logloss_multinomial
    assert 0.9 < ratio and ratio < 1.9

    logloss_ovr = log_loss(y_test, ovr.predict_proba(X_test))

    assert math.isclose(logloss_binary, logloss_ovr, rel_tol=1e-5)

    original = ovr.copy()._multinomialize()

    logloss_original = log_loss(y_test, original.predict_proba(X_test))

    ratio2 = logloss_original / logloss_multinomial
    assert 0.75 < ratio and ratio < 1.75


def test_monotonize():
    data = synthetic_classification()
    X = data["full"]["X"]
    y = data["full"]["y"]

    clf = ExplainableBoostingClassifier()
    clf.fit(X, y)

    intercept = clf.intercept_
    clf.monotonize(0, increasing=True)
    clf.monotonize(1, increasing=False)
    clf.monotonize(2, increasing="auto")
    clf.monotonize(3)

    valid_ebm(clf)

    assert abs(np.average(clf.term_scores_[0], weights=clf.bin_weights_[0])) < 0.0001
    assert abs(np.average(clf.term_scores_[1], weights=clf.bin_weights_[1])) < 0.0001
    assert abs(np.average(clf.term_scores_[2], weights=clf.bin_weights_[2])) < 0.0001
    assert abs(np.average(clf.term_scores_[3], weights=clf.bin_weights_[3])) < 0.0001
    assert np.all(np.diff(clf.term_features_[0]) >= 0)
    assert np.all(np.diff(clf.term_features_[1]) <= 0)
    diff = np.diff(clf.term_features_[2])
    assert np.all(diff >= 0) or np.all(diff <= 0)
    assert intercept == clf.intercept_


def test_ebm_remove_features():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]

    clf = ExplainableBoostingRegressor(interactions=[(1, 2), (1, 3)])
    clf.fit(X, y)
    clf.remove_features(["A", 2])

    assert clf.feature_names_in_ == ["B", "D"]
    assert len(clf.histogram_edges_) == 2
    assert len(clf.histogram_weights_) == 2
    assert len(clf.unique_val_counts_) == 2
    assert len(clf.bins_) == 2
    assert len(clf.feature_names_in_) == 2
    assert len(clf.feature_types_in_) == 2
    assert clf.feature_bounds_.shape[0] == 2
    assert clf.n_features_in_ == 2

    assert clf.term_names_ == ["B", "D", "B & D"]
    assert len(clf.term_features_) == 3
    assert len(clf.term_scores_) == 3
    assert len(clf.bagged_scores_) == 3
    assert len(clf.bagged_intercept_) == len(clf.bagged_scores_[0])
    assert len(clf.standard_deviations_) == 3
    assert len(clf.bin_weights_) == 3


def test_ebm_sweep():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]

    clf = ExplainableBoostingRegressor(interactions=[(1, 2), (1, 3)])
    clf.fit(X, y)

    clf.term_scores_[0].fill(0)  # set 'A' to zero
    clf.term_scores_[len(clf.term_scores_) - 1].fill(0)  # set 'B & D' to zero

    # the synthetic dataset is random and if no progress is made
    # during boosting it can have zeros, so force to non-zero
    clf.term_scores_[1][1] = 99.0
    clf.term_scores_[2][1] = 99.0
    clf.term_scores_[3][1] = 99.0
    clf.term_scores_[4][1, 1] = 99.0

    clf.sweep(terms=True, bins=True, features=True)

    # check that sweeping the features removed feature 'A' which was not used in a term
    assert clf.feature_names_in_ == ["B", "C", "D"]
    assert len(clf.histogram_edges_) == 3
    assert len(clf.histogram_weights_) == 3
    assert len(clf.unique_val_counts_) == 3
    assert len(clf.bins_) == 3
    assert len(clf.feature_names_in_) == 3
    assert len(clf.feature_types_in_) == 3
    assert clf.feature_bounds_.shape[0] == 3
    assert clf.n_features_in_ == 3

    # check that sweeping the bins deleted the pair bins in D
    assert len(clf.bins_[0]) == 2  # feature 'B'
    assert len(clf.bins_[1]) == 2  # feature 'C'
    assert len(clf.bins_[2]) == 1  # feature 'D'

    # check that sweeping the terms removed the zeroed terms
    assert clf.term_names_ == ["B", "C", "D", "B & C"]
    assert len(clf.term_features_) == 4
    assert len(clf.term_scores_) == 4
    assert len(clf.bagged_scores_) == 4
    assert len(clf.bagged_intercept_) == len(clf.bagged_scores_[0])
    assert len(clf.standard_deviations_) == 4
    assert len(clf.bin_weights_) == 4


def test_copy():
    data = synthetic_classification()
    X = data["full"]["X"]
    y = data["full"]["y"]

    clf = ExplainableBoostingClassifier()
    clf.fit(X, y)

    ebm_copy = clf.copy()
    clf.term_scores_ = None  # make the original invalid

    valid_ebm(ebm_copy)


@pytest.mark.slow
def test_unknown_multiclass_category():
    data = iris_classification()
    X_train = data["train"]["X"]
    y_train = data["train"]["y"]

    X_test = data["test"]["X"]

    # Add categorical feature
    X_train["cat_feature"] = [
        np.random.choice(["a", "b", "c"]) for x in range(X_train.shape[0])
    ]
    X_test["cat_feature"] = [
        "d" for x in range(X_test.shape[0])
    ]  # Unknown category in test set

    # X_train['cat_feature'][1] = np.nan
    # X_test['cat_feature'][1] = np.nan

    clf = ExplainableBoostingClassifier(interactions=0)
    clf.fit(X_train, y_train)

    # Term contributions for categorical feature should always be 0 in test
    assert np.all(clf.explain_local(X_train).data(0)["scores"][-1] != 0)
    assert np.all(clf.explain_local(X_test).data(0)["scores"][-1] == 0)


@pytest.mark.slow
def test_unknown_binary_category():
    X, y, names, types = make_synthetic(classes=2, output_type="float")

    ebm = ExplainableBoostingClassifier(
        names, types, interactions=[[0, -1], [1, 2], [-1, 3]]
    )
    ebm.fit(X, y)

    orig = ebm.eval_terms(X[0])

    X[0, -1] = -9.9  # unseen categorical
    unseen = ebm.eval_terms(X[0])

    assert unseen[0, ebm.n_features_in_ - 1] == 0
    assert unseen[0, -3] == 0
    assert unseen[0, -1] == 0

    # overwrite the 3 values that should be zero with the orig scores
    unseen[0, ebm.n_features_in_ - 1] = orig[0, ebm.n_features_in_ - 1]
    unseen[0, -3] = orig[0, -3]
    unseen[0, -1] = orig[0, -1]

    # everything else should be unchanged other than the zeroed values we overwrote
    assert np.array_equal(orig, unseen)


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

    clf = ExplainableBoostingClassifier(n_jobs=-2, interactions=0, outer_bags=2)
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
    assert clf_global.data(2)["scores"][-1][-1] > 3.5
    assert clf_global.data(2)["scores"][1][1] > 3.5


def test_ebm_tripple():
    data = iris_classification()
    X_train = data["train"]["X"]
    y_train = data["train"]["y"]

    X_test = data["test"]["X"]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            "Interactions with 3 or more terms are not graphed in global explanations.*",
        )

        # iris is multiclass, but for now pretend this is a regression problem
        clf = ExplainableBoostingRegressor(
            interactions=[(0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 1)]
        )
        clf.fit(X_train, y_train)
        clf.predict(X_test)
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

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Missing values detected.*")

        clf = ExplainableBoostingRegressor(n_jobs=-2, interactions=0)
        clf.fit(X, y)
        clf.predict(X)

        valid_ebm(clf)


def test_ebm_only_missing():
    X = np.full((10, 10), np.nan)
    y = np.full(10, 0)
    y[0] = 1

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Missing values detected.*")

        clf = ExplainableBoostingClassifier(n_jobs=1)
        clf.fit(X, y)
        clf.predict(X)
        clf.explain_global()
        clf.explain_local(X, y)


def test_ebm_synthetic_singleclass_classification():
    X, y, names, types = make_synthetic(classes=2, output_type="float")
    y[:] = 0

    clf = ExplainableBoostingClassifier(names, types)
    clf.fit(X, y)

    assert clf.link_ == "monoclassification"
    assert clf.term_scores_[0][1] == -np.inf
    assert clf.intercept_[0] == -np.inf
    assert clf.bagged_scores_[0][0, 1] == -np.inf
    assert clf.bagged_intercept_[0] == -np.inf

    prob_scores = clf.predict_proba(X)
    assert prob_scores.ndim == 2
    assert prob_scores.shape[0] == len(y)
    assert prob_scores.shape[1] == 1
    assert (prob_scores == 1.0).all()

    predicts = clf.predict(X)
    assert predicts.ndim == 1
    assert predicts.shape[0] == len(y)
    assert not np.any(predicts)

    scores = clf._predict_score(X)
    assert scores.ndim == 1
    assert len(scores) == len(y)

    explanations = clf.eval_terms(X)
    assert explanations.ndim == 2
    assert explanations.shape[0] == len(y)
    assert explanations.shape[1] == len(clf.term_features_)


@pytest.mark.visual
@pytest.mark.slow
def test_ebm_uniform():
    # TODO: expand this test to use the other feature types available
    #       and evaluate using an feature value outside of the training range
    from sklearn.metrics import roc_auc_score  # type: ignore

    X, y, names, types = make_synthetic(classes=2, output_type="float")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20)

    types[0] = "uniform"

    clf = ExplainableBoostingClassifier(names, types)
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

    smoke_test_explanations(global_exp, local_exp, 6000)


@pytest.mark.visual
@pytest.mark.slow
def test_ebm_uniform_multiclass():
    data = iris_classification()
    X_train = data["train"]["X"]
    y_train = data["train"]["y"]

    X_test = data["test"]["X"]
    y_test = data["test"]["y"]

    feature_types = [None] * X_train.shape[1]
    feature_types[0] = "uniform"

    clf = ExplainableBoostingClassifier(interactions=0, feature_types=feature_types)
    clf.fit(X_train, y_train)

    assert accuracy_score(y_test, clf.predict(X_test)) > 0.9

    global_exp = clf.explain_global()
    local_exp = clf.explain_local(X_test, y_test)

    smoke_test_explanations(global_exp, local_exp, 6001)


@pytest.mark.visual
@pytest.mark.slow
def test_ebm_binary():
    from sklearn.metrics import roc_auc_score  # type: ignore

    X, y, names, types = make_synthetic(classes=2, output_type="float")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20)

    clf = ExplainableBoostingClassifier(names, types)
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

    smoke_test_explanations(global_exp, local_exp, 6000)


def test_eval_terms_regression():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]

    clf = ExplainableBoostingRegressor()
    clf.fit(X, y)

    explanations = clf.eval_terms(X)

    scores = explanations.sum(axis=1) + clf.intercept_
    assert np.allclose(clf.predict(X), scores)

    # for RMSE with identity link, the scores are the predictions
    predictions = inv_link(scores, clf.link_, clf.link_param_)
    assert np.allclose(clf.predict(X), predictions)


def test_eval_terms_binary():
    data = synthetic_classification()
    X = data["train"]["X"]
    y = data["train"]["y"]

    clf = ExplainableBoostingClassifier()
    clf.fit(X, y)

    explanations = clf.eval_terms(X)

    scores = explanations.sum(axis=1) + clf.intercept_
    assert np.allclose(clf._predict_score(X), scores)

    probabilities = inv_link(scores, clf.link_, clf.link_param_)
    assert np.allclose(clf.predict_proba(X), probabilities)


def test_eval_terms_multiclass():
    data = synthetic_multiclass()
    X = data["train"]["X"]
    y = data["train"]["y"]

    clf = ExplainableBoostingClassifier(interactions=0)
    clf.fit(X, y)

    explanations = clf.eval_terms(X)

    scores = explanations.sum(axis=1) + clf.intercept_
    assert np.allclose(clf._predict_score(X), scores)

    probabilities = inv_link(scores, clf.link_, clf.link_param_)
    assert np.allclose(clf.predict_proba(X), probabilities)


def test_ebm_sample_weight():
    X, y, names, types = make_synthetic(classes=2, output_type="float")

    ebm = ExplainableBoostingClassifier(names, types)
    ebm.fit(X, y)

    weights = np.full(len(y), 128.0)
    weighted = ExplainableBoostingClassifier(names, types)
    weighted.fit(X, y, sample_weight=weights)

    # if all the weights are identical the models should be identical
    assert np.array_equal(ebm.predict_proba(X), weighted.predict_proba(X))

    # change a single weight
    weights[0] = 1.1
    changed = ExplainableBoostingClassifier(names, types)
    changed.fit(X, y, sample_weight=weights)

    assert not np.array_equal(ebm.predict_proba(X), changed.predict_proba(X))


@pytest.mark.visual
@pytest.mark.slow
def test_ebm_iris():
    data = iris_classification()
    X_train = data["train"]["X"]
    y_train = data["train"]["y"]

    X_test = data["test"]["X"]
    y_test = data["test"]["y"]

    clf = ExplainableBoostingClassifier(interactions=0)
    clf.fit(X_train, y_train)

    assert accuracy_score(y_test, clf.predict(X_test)) > 0.9

    global_exp = clf.explain_global()
    local_exp = clf.explain_local(X_test, y_test)

    smoke_test_explanations(global_exp, local_exp, 6001)


@pytest.mark.visual
@pytest.mark.slow
def test_ebm_sparse():
    """Validate running EBM on scipy sparse data"""
    from sklearn.datasets import make_multilabel_classification  # type: ignore

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
    smoke_test_explanations(global_exp, local_exp, 6002)


@pytest.mark.slow
def test_zero_validation():
    data = synthetic_classification()
    X = data["full"]["X"]
    y = data["full"]["y"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "If validation_size is 0*")

        clf = ExplainableBoostingClassifier(n_jobs=1, interactions=2, validation_size=0)
        clf.fit(X, y)


@pytest.mark.visual
@pytest.mark.slow
def test_dp_ebm_binary():
    from sklearn.metrics import roc_auc_score  # type: ignore

    X, y, names, types = make_synthetic(classes=2, n_samples=10000, output_type="float")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20)

    w_tr = np.ones_like(y_tr)
    w_tr[-1] = 2

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Possible privacy violation.*")

        clf = DPExplainableBoostingClassifier(names, types, epsilon=1)
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

        smoke_test_explanations(global_exp, local_exp, 6000)


def test_dp_ebm_synthetic_regression():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]
    w = np.ones_like(y)
    w[-1] = 2

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Possible privacy violation.*")

        clf = DPExplainableBoostingRegressor()
        clf.fit(X, y, w)
        clf.predict(X)

        valid_ebm(clf)


def test_dp_ebm_external_privacy_bounds():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]

    # synthetic regression is all sampled from N(0, 1)
    privacy_bounds = {0: (-3, 3), 1: (-3, 3), 2: (-3, 3), 3: (-3, 3)}

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Possible privacy violation*")

        clf = DPExplainableBoostingRegressor(
            privacy_bounds=privacy_bounds, privacy_target_min=-3, privacy_target_max=3
        )
        clf.fit(X, y)
        clf.predict(X)

        valid_ebm(clf)


@pytest.mark.slow
def test_ebm_calibrated_classifier_cv():
    """Tests if unsigned integers can be handled when
    using CalibratedClassifierCV.
    """
    from sklearn.calibration import CalibratedClassifierCV  # type: ignore

    X = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.uint8,
    )

    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.uint8)

    clf = ExplainableBoostingClassifier()
    calib = CalibratedClassifierCV(clf)
    calib.fit(X, y)


def test_ebm_unknown_value_at_predict():
    """Tests if unsigned integers can be handled when unknown values
    are found by predict.

    e.g. feature 3 has only 0's in X but a 1 in X_test.
    """
    X = np.array(
        [[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
        dtype=np.uint8,
    )

    X_test = np.array([[0, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 0]], dtype=np.uint8)

    y = np.array([0, 1, 1, 1, 1], dtype=np.uint8)

    clf = ExplainableBoostingClassifier()
    clf.fit(X, y)
    clf.predict(X_test)

    valid_ebm(clf)


def test_bags():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]

    n_samples = X.shape[0]

    init_score = np.full(n_samples, 3.0)
    sample_weight = np.full(n_samples, 99.0)

    bags = np.zeros((1, n_samples), np.int8)
    for idx in range(n_samples - 2, -1, -2):
        bags[0, idx] = 1
        init_score[idx] = -0.5 * idx - 0.25
        sample_weight[idx] = 0.25 * idx + 0.5

    keep = bags[0] != 0

    X0 = X[keep]
    y0 = y[keep]
    init_score0 = init_score[keep]
    sample_weight0 = sample_weight[keep]

    clf = ExplainableBoostingRegressor(
        max_bins=n_samples + 2,
        max_interaction_bins=n_samples + 2,
        max_rounds=100,
        outer_bags=1,
        validation_size=0,
        smoothing_rounds=0,
        interaction_smoothing_rounds=0,
    )
    clf.fit(X0, y0, sample_weight=sample_weight0, init_score=init_score0)
    pred0 = clf.predict(X0)

    clf = ExplainableBoostingRegressor(
        max_bins=n_samples + 2,
        max_interaction_bins=n_samples + 2,
        max_rounds=100,
        outer_bags=1,
        validation_size=0,
        smoothing_rounds=0,
        interaction_smoothing_rounds=0,
    )
    clf.fit(X, y, sample_weight=sample_weight, bags=bags, init_score=init_score)
    pred1 = clf.predict(X0)

    assert np.allclose(pred0, pred1)


@pytest.mark.skip(
    reason="can't run this test reliably until we depend on scikit-learn 0.22"
)
def test_scikit_learn_compatibility():
    """Run scikit-learn compatibility tests"""

    # sklearn tests in:
    # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/estimator_checks.py

    skip_tests = {
        "check_dtype_object",  # the error message required to pass is too specific and incorrect for us
        "check_classifiers_one_label",  # TODO: fix this!  We should accept 1 category
        "check_classifiers_regression_target",  # we're more permissive and convert any y values to str
        "check_supervised_y_no_nan",  # error message too specific
        "check_supervised_y_2d",  # we ignore useless added dimensions
        "check_fit2d_predict1d",  # we accept 1d for predict
        "check_fit2d_1sample",  # TODO: we allow fitting on 1 sample, but this kind of input is likely a bug from the caller, so change this
    }
    for estimator, check_func in check_estimator(
        ExplainableBoostingClassifier(), generate_only=True
    ):
        f = check_func.func
        module = f.__module__
        shortname = f.__name__
        fullname = f"{module}.{shortname}"
        if shortname not in skip_tests:
            try:
                check_func(estimator)
            except BaseException as e:
                # print(fullname)
                # print(f"{type(e).__name__}: {e}")
                # print()
                pass

    for estimator, check_func in check_estimator(
        ExplainableBoostingRegressor(), generate_only=True
    ):
        f = check_func.func
        module = f.__module__
        shortname = f.__name__
        fullname = f"{module}.{shortname}"
        if shortname not in skip_tests:
            try:
                check_func(estimator)
            except BaseException as e:
                # print(fullname)
                # print(f"{type(e).__name__}: {e}")
                # print()
                pass

    for estimator, check_func in check_estimator(
        DPExplainableBoostingClassifier(), generate_only=True
    ):
        f = check_func.func
        module = f.__module__
        shortname = f.__name__
        fullname = f"{module}.{shortname}"
        if shortname not in skip_tests:
            try:
                check_func(estimator)
            except BaseException as e:
                # print(fullname)
                # print(f"{type(e).__name__}: {e}")
                # print()
                pass

    for estimator, check_func in check_estimator(
        DPExplainableBoostingRegressor(), generate_only=True
    ):
        f = check_func.func
        module = f.__module__
        shortname = f.__name__
        fullname = f"{module}.{shortname}"
        if shortname not in skip_tests:
            try:
                check_func(estimator)
            except BaseException as e:
                # print(fullname)
                # print(f"{type(e).__name__}: {e}")
                # print()
                pass


def test_json_classification():
    data = synthetic_classification()
    X = data["full"]["X"]
    y = data["full"]["y"]

    X["A"] = pd.cut(
        X["A"],
        [-np.inf, -1, -0.5, 0.5, 1, np.inf],
        labels=["apples", "oranges", "0", "almonds", "peanuts"],
        ordered=False,
    )
    X["B"] = pd.cut(
        X["B"],
        [-np.inf, -0.5, 0.5, np.inf],
        labels=["low", "medium", "high"],
        ordered=True,
    )
    X["C"] = X["C"].mask(X["C"] < 0, 0)

    clf = ExplainableBoostingClassifier(
        outer_bags=3, max_bins=10, max_interaction_bins=4, interactions=[(1, 2), (2, 0)]
    )
    clf.fit(X, y)

    # combine the last two bins into one bin
    max_val = max(clf.bins_[0][0].values())
    clf.bins_[0][0] = {
        key: value if value != max_val else max_val - 1
        for key, value in clf.bins_[0][0].items()
    }

    clf.term_scores_[0] = np.delete(clf.term_scores_[0], 1)
    clf.term_scores_[0][1] = -np.inf
    clf.term_scores_[0][2] = np.inf
    clf.term_scores_[0][3] = np.nan

    clf.standard_deviations_[0] = np.delete(clf.standard_deviations_[0], 1)
    clf.bagged_scores_[0] = np.array(
        [
            np.delete(clf.bagged_scores_[0][0], 1),
            np.delete(clf.bagged_scores_[0][1], 1),
            np.delete(clf.bagged_scores_[0][2], 1),
        ]
    )

    clf.bin_weights_[0] = np.delete(clf.bin_weights_[0], 1)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "JSON formats are in beta.*")

        clf.to_jsonable(detail="all")


def test_json_multiclass():
    data = synthetic_multiclass()
    X = data["full"]["X"]
    y = data["full"]["y"]
    feature_types = ["continuous"] * X.shape[1]
    feature_types[0] = "nominal"
    clf = ExplainableBoostingClassifier(
        max_bins=10, feature_types=feature_types, interactions=0
    )
    clf.fit(X, y)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "JSON formats are in beta.*")

        clf.to_jsonable(detail="all")


def test_json_regression():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]
    feature_types = ["continuous"] * X.shape[1]
    feature_types[0] = "nominal"

    clf = ExplainableBoostingRegressor(
        max_bins=5,
        max_interaction_bins=4,
        feature_types=feature_types,
        interactions=[(1, 2), (2, 3)],
    )
    clf.fit(X, y)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "JSON formats are in beta.*")

        clf.to_jsonable(detail="all")


def test_json_dp_classification():
    data = synthetic_classification()
    X = data["full"]["X"]
    y = data["full"]["y"]
    feature_types = ["continuous"] * X.shape[1]
    feature_types[0] = "nominal"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Possible privacy violation*")
        warnings.filterwarnings("ignore", "JSON formats are in beta.*")

        clf = DPExplainableBoostingClassifier(max_bins=10, feature_types=feature_types)
        clf.fit(X, y)
        clf.term_scores_[0][0] = np.nan
        clf.term_scores_[0][1] = np.inf
        clf.term_scores_[0][2] = -np.inf
        clf.to_jsonable(detail="all")


def test_json_dp_regression():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]
    feature_types = ["continuous"] * X.shape[1]
    feature_types[0] = "nominal"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Possible privacy violation*")
        warnings.filterwarnings("ignore", "JSON formats are in beta.*")

        clf = DPExplainableBoostingRegressor(max_bins=5, feature_types=feature_types)
        clf.fit(X, y)
        clf.to_jsonable(detail="all")


def test_to_json():
    data = synthetic_classification()
    X = data["full"]["X"]
    y = data["full"]["y"]

    clf = ExplainableBoostingClassifier()
    clf.fit(X, y)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "JSON formats are in beta.*")

        file_like_string_writer = StringIO()
        jsonable = clf.to_json(file_like_string_writer)
        json_data = file_like_string_writer.getvalue()
        jsonable = json.loads(json_data)
        assert "ebm" in jsonable


def test_exclude_explicit():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]

    clf = ExplainableBoostingRegressor(
        interactions=[(1, 2), (2, 3)], exclude=[(3, 2), "B", (2,)]
    )
    clf.fit(X, y)
    clf.predict(X)

    assert (2, 3) not in clf.term_features_
    assert (1,) not in clf.term_features_
    assert (2,) not in clf.term_features_
    assert (0,) in clf.term_features_
    assert (3,) in clf.term_features_

    valid_ebm(clf)


def test_exclude_implicit():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]

    clf = ExplainableBoostingRegressor(
        interactions=99999999, exclude=[(3, "C"), 1, (2,)]
    )
    clf.fit(X, y)
    clf.predict(X)

    assert (2, 3) not in clf.term_features_
    assert (1,) not in clf.term_features_
    assert (2,) not in clf.term_features_
    assert (0,) in clf.term_features_
    assert (3,) in clf.term_features_

    valid_ebm(clf)


def test_exclude_complete_feature():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]

    clf = ExplainableBoostingRegressor(interactions=[], exclude=[0, 1])
    clf.fit(X, y)
    clf.predict(X)

    assert (0,) not in clf.term_features_
    assert (1,) not in clf.term_features_
    assert (2,) in clf.term_features_
    assert (3,) in clf.term_features_


def test_exclude_all():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]

    clf = ExplainableBoostingRegressor(interactions=[], exclude="mains")
    clf.fit(X, y)
    clf.predict(X)

    assert len(clf.term_features_) == 0


def test_ebm_remove_terms():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]

    clf = ExplainableBoostingRegressor(n_jobs=-2, interactions=0)
    clf.fit(X, y)
    assert clf.term_names_ == ["A", "B", "C", "D"]
    clf.remove_terms(["A", "C"])
    assert clf.term_names_ == ["B", "D"]
    assert len(clf.term_features_) == 2
    assert len(clf.term_names_) == 2
    assert len(clf.term_scores_) == 2
    assert len(clf.bagged_scores_) == 2
    assert len(clf.bagged_intercept_) == len(clf.bagged_scores_[0])
    assert len(clf.standard_deviations_) == 2
    assert len(clf.bin_weights_) == 2


def test_ebm_scale():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]

    clf = ExplainableBoostingRegressor(n_jobs=-2, interactions=0)
    clf.fit(X, y)
    assert clf.term_names_ == ["A", "B", "C", "D"]

    # the synthetic dataset is random and if no progress is made
    # during boosting it can have zeros, so force to non-zero
    clf.term_scores_[0][1] = 99.0
    clf.term_scores_[1][1] = 99.0
    clf.term_scores_[2][1] = 99.0
    clf.term_scores_[3][1] = 99.0

    # The following is equivalent to calling `clf.remove_terms(["A", "C"])`
    clf.scale(0, factor=0)
    clf.scale("B", factor=0.5)
    clf.scale("C", factor=0)
    clf.scale(3, factor=2.0)
    clf.sweep()
    assert clf.term_names_ == ["B", "D"]
    assert len(clf.term_features_) == 2
    assert len(clf.term_names_) == 2
    assert len(clf.term_scores_) == 2
    assert len(clf.bagged_scores_) == 2
    assert len(clf.bagged_intercept_) == len(clf.bagged_scores_[0])
    assert len(clf.standard_deviations_) == 2
    assert len(clf.bin_weights_) == 2
