# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..linear import LogisticRegression, LinearRegression
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.linear_model import LogisticRegression as SKLogistic
from sklearn.linear_model import Lasso as SKLinear
import numpy as np


def test_linear_regression():
    dataset = load_diabetes()
    X, y = dataset.data, dataset.target
    feature_names = dataset.feature_names

    sk_lr = SKLinear(random_state=1)
    our_lr = LinearRegression(feature_names=feature_names, random_state=1)

    sk_lr.fit(X, y)
    our_lr.fit(X, y)

    sk_pred = sk_lr.predict(X)
    our_pred = our_lr.predict(X)
    assert np.allclose(sk_pred, our_pred)

    # With labels
    local_expl = our_lr.explain_local(X, y)
    local_viz = local_expl.visualize(0)
    assert local_viz is not None

    # Without labels
    local_expl = our_lr.explain_local(X)
    local_viz = local_expl.visualize(0)
    assert local_viz is not None

    global_expl = our_lr.explain_global()
    global_viz = global_expl.visualize()
    assert global_viz is not None


def test_logistic_regression():
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    feature_names = cancer.feature_names

    sk_lr = SKLogistic(tol=0.01, random_state=1)
    our_lr = LogisticRegression(tol=0.01, feature_names=feature_names, random_state=1)

    sk_lr.fit(X, y)
    our_lr.fit(X, y)

    sk_pred = sk_lr.predict_proba(X)
    our_pred = our_lr.predict_proba(X)
    assert np.allclose(sk_pred, our_pred)

    sk_pred = sk_lr.predict(X)
    our_pred = our_lr.predict(X)
    assert np.allclose(sk_pred, our_pred)

    # With labels
    local_expl = our_lr.explain_local(X, y)
    local_viz = local_expl.visualize(0)
    assert local_viz is not None

    # Without labels
    local_expl = our_lr.explain_local(X)
    local_viz = local_expl.visualize(0)
    assert local_viz is not None

    global_expl = our_lr.explain_global()
    global_viz = global_expl.visualize()
    assert global_viz is not None


def test_sorting():
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    feature_names = cancer.feature_names

    sk_lr = SKLogistic(tol=0.01, random_state=1)
    our_lr = LogisticRegression(tol=0.01, feature_names=feature_names, random_state=1)

    sk_lr.fit(X, y)
    our_lr.fit(X, y)

    sk_pred = sk_lr.predict_proba(X)
    our_pred = our_lr.predict_proba(X)
    assert np.allclose(sk_pred, our_pred)

    sk_pred = sk_lr.predict(X)
    our_pred = our_lr.predict(X)
    assert np.allclose(sk_pred, our_pred)

    # With labels
    local_expl = our_lr.explain_local(X, y)
    local_viz = local_expl.visualize(0)
    assert local_viz is not None

    # Without labels
    local_expl = our_lr.explain_local(X)
    local_viz = local_expl.visualize(0)
    assert local_viz is not None

    global_expl = our_lr.explain_global()
    global_viz = global_expl.visualize()
    assert global_viz is not None

    from ...visual.plot import sort_take, mli_sort_take, get_sort_indexes

    data_dict = sort_take(
        global_expl.data(), sort_fn=lambda x: -abs(x), top_n=15, reverse_results=True
    )
    scores_data = global_expl.data(-1)["mli"][0]["value"]["scores"]
    sort_indexes = get_sort_indexes(scores_data, sort_fn=lambda x: -abs(x), top_n=15)
    sorted_scores = mli_sort_take(scores_data, sort_indexes, reverse_results=True)
    assert data_dict["scores"] == sorted_scores
