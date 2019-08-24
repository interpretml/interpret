# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..decisiontree import ClassificationTree, RegressionTree
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.tree import DecisionTreeClassifier as SKDT
from sklearn.tree import DecisionTreeRegressor as SKRT
import numpy as np


def test_rt():
    boston = load_boston()
    X, y = boston.data, boston.target
    feature_names = boston.feature_names

    sk_dt = SKRT(random_state=1, max_depth=3)
    our_dt = RegressionTree(feature_names=feature_names, random_state=1)

    sk_dt.fit(X, y)
    our_dt.fit(X, y)

    sk_pred = sk_dt.predict(X)
    our_pred = our_dt.predict(X)
    assert np.allclose(sk_pred, our_pred)

    # With labels
    local_expl = our_dt.explain_local(X, y)
    local_viz = local_expl.visualize(0)
    assert local_viz is not None

    # Without labels
    local_expl = our_dt.explain_local(X)
    local_viz = local_expl.visualize(0)
    assert local_viz is not None

    global_expl = our_dt.explain_global()
    global_viz = global_expl.visualize()
    assert global_viz is not None


def test_dt():
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    feature_names = cancer.feature_names

    sk_dt = SKDT(random_state=1, max_depth=3)
    our_dt = ClassificationTree(feature_names=feature_names, random_state=1)

    sk_dt.fit(X, y)
    our_dt.fit(X, y)

    sk_pred = sk_dt.predict_proba(X)
    our_pred = our_dt.predict_proba(X)
    assert np.allclose(sk_pred, our_pred)

    sk_pred = sk_dt.predict(X)
    our_pred = our_dt.predict(X)
    assert np.allclose(sk_pred, our_pred)

    # With labels
    local_expl = our_dt.explain_local(X, y)
    local_viz = local_expl.visualize(0)
    assert local_viz is not None

    # Without labels
    local_expl = our_dt.explain_local(X)
    local_viz = local_expl.visualize(0)
    assert local_viz is not None

    global_expl = our_dt.explain_global()
    global_viz = global_expl.visualize()
    assert global_viz is not None
