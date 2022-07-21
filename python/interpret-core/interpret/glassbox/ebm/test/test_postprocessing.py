# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import pytest

from ..postprocessing import multiclass_postprocess

from ....test.utils import (
    synthetic_multiclass,
    synthetic_classification,
    adult_classification,
    iris_classification,
)

from sklearn.model_selection import (
    cross_validate,
    StratifiedShuffleSplit,
    train_test_split,
)
from ..ebm import ExplainableBoostingRegressor, ExplainableBoostingClassifier
from ..utils import EBMUtils, merge_ebms

import numpy as np


def test_multiclass_postprocess_smoke():
    n = 1000
    d = 2
    k = 3
    b = 10

    X_binned = np.random.randint(b, size=(d, n))
    feature_graphs = []
    for _ in range(d):
        feature_graphs.append(np.random.rand(b, k))

    def binned_predict_proba(X_binned, k=3):
        n = X_binned.shape[1]
        return 1 / k * np.ones((n, k))

    feature_types = ["numeric"] * d
    results = multiclass_postprocess(
        X_binned, feature_graphs, binned_predict_proba, feature_types
    )

    assert "intercepts" in results
    assert "feature_graphs" in results

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

def test_merge_ebms():
    
    data = adult_classification()
    X = data["full"]["X"]
    y = data["full"]["y"]
    X_te = data["test"]["X"]
    y_te = data["test"]["y"]   
    
    random_state =1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state)
    X_train[:, 4] = "-7"
    X_train[0, 4] = "8"
    X_train[1, 4] = "5.0"
    X_train[2, 4] = "+5"
    # make this categorical to merge with the continuous feature in the other ebms
    X_train[3, 4] = "me" 
    X_train[4, 4] = "you"
    ebm1 = ExplainableBoostingClassifier(random_state=random_state, n_jobs=-1, max_interaction_bins=3, interactions=[(8,4,0)])
    ebm1.fit(X_train, y_train)  

    random_state +=10
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=random_state)
    ebm2 = ExplainableBoostingClassifier(random_state=random_state, n_jobs=-1, max_interaction_bins=4, interactions=[(8, 2), (10, 11), (12, 7)])
    ebm2.fit(X_train, y_train)  

    random_state +=10
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.60, random_state=random_state)
    ebm3 = ExplainableBoostingClassifier(random_state=random_state, n_jobs=-1, max_interaction_bins=5, interactions=[(12, 7), (2, 8)])
    ebm3.fit(X_train, y_train) 
        
    merged_ebm1 = merge_ebms([ebm1, ebm2 , ebm3])
    valid_ebm(merged_ebm1)
    global_exp = merged_ebm1.explain_global()
    local_exp = merged_ebm1.explain_local(X_te[:5, :], y_te[:5])
    _smoke_test_explanations(global_exp, local_exp, 6000)

    random_state +=10
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=random_state)
    ebm4 = ExplainableBoostingClassifier(random_state=random_state, n_jobs=-1, max_interaction_bins=8, interactions=2)
    ebm4.fit(X_train, y_train) 
        
    merged_ebm2 = merge_ebms([merged_ebm1, ebm4])
    valid_ebm(merged_ebm2)
    global_exp = merged_ebm2.explain_global()
    local_exp = merged_ebm2.explain_local(X_te[:5, :], y_te[:5])
    _smoke_test_explanations(global_exp, local_exp, 6000)

    random_state +=10
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=random_state)
    ebm5 = ExplainableBoostingClassifier(random_state=random_state, n_jobs=-1, max_interaction_bins=8, interactions=2)
    ebm5.fit(X_train, y_train) 
        
    merged_ebm3 = merge_ebms([ebm5, merged_ebm2])
    valid_ebm(merged_ebm3)
    global_exp = merged_ebm3.explain_global()
    local_exp = merged_ebm3.explain_local(X_te[:5, :], y_te[:5])
    _smoke_test_explanations(global_exp, local_exp, 6000)
