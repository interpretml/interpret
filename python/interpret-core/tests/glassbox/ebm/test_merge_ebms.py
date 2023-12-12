# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from ...tutils import (
    adult_classification,
    iris_classification,
    smoke_test_explanations,
)

from sklearn.model_selection import train_test_split
from interpret.glassbox import ExplainableBoostingClassifier, merge_ebms

import numpy as np
import warnings

def valid_ebm(ebm):
    assert ebm.term_features_[0] == (0,)

    for term_scores in ebm.term_scores_:
        all_finite = np.isfinite(term_scores).all()
        assert all_finite


def test_merge_ebms():
    data = adult_classification()
    X = data["full"]["X"]
    y = data["full"]["y"]
    X_te = data["test"]["X"]
    y_te = data["test"]["y"]

    random_state = 1
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=random_state
    )
    X_train[:, 4] = "-7"
    X_train[0, 4] = "8"
    X_train[1, 4] = "5.0"
    X_train[2, 4] = "+5"
    # make this categorical to merge with the continuous feature in the other ebms
    X_train[3, 4] = "me"
    X_train[4, 4] = "you"

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Dropping term.*') 
        warnings.filterwarnings('ignore', 'Interactions with 3 or more terms are not graphed in global explanations.*') 

        ebm1 = ExplainableBoostingClassifier(
            random_state=random_state,
            max_interaction_bins=3,
            interactions=[(8, 4, 0)],
        )
        ebm1.fit(X_train, y_train)

        random_state += 10
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.40, random_state=random_state
        )
        ebm2 = ExplainableBoostingClassifier(
            random_state=random_state,
            max_interaction_bins=4,
            interactions=[(8, 2), (10, 11), (12, 7)],
        )
        ebm2.fit(X_train, y_train)

        random_state += 10
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.60, random_state=random_state
        )
        ebm3 = ExplainableBoostingClassifier(
            random_state=random_state,
            max_interaction_bins=5,
            interactions=[(12, 7), (2, 8)],
        )
        ebm3.fit(X_train, y_train)

        merged_ebm1 = merge_ebms([ebm1, ebm2, ebm3])
        valid_ebm(merged_ebm1)
        global_exp = merged_ebm1.explain_global()
        local_exp = merged_ebm1.explain_local(X_te[:5, :], y_te[:5])
        smoke_test_explanations(global_exp, local_exp, 6000)

        random_state += 10
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.10, random_state=random_state
        )
        ebm4 = ExplainableBoostingClassifier(
            random_state=random_state, max_interaction_bins=8, interactions=2
        )
        ebm4.fit(X_train, y_train)

        merged_ebm2 = merge_ebms([merged_ebm1, ebm4])
        valid_ebm(merged_ebm2)
        global_exp = merged_ebm2.explain_global()
        local_exp = merged_ebm2.explain_local(X_te[:5, :], y_te[:5])
        smoke_test_explanations(global_exp, local_exp, 6000)

        random_state += 10
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.50, random_state=random_state
        )
        ebm5 = ExplainableBoostingClassifier(
            random_state=random_state, max_interaction_bins=8, interactions=2
        )
        ebm5.fit(X_train, y_train)

        merged_ebm3 = merge_ebms([ebm5, merged_ebm2])
        valid_ebm(merged_ebm3)
        global_exp = merged_ebm3.explain_global()
        local_exp = merged_ebm3.explain_local(X_te[:5, :], y_te[:5])
        smoke_test_explanations(global_exp, local_exp, 6000)


def test_merge_ebms_multiclass():
    data = iris_classification()
    X = data["full"]["X"]
    y = data["full"]["y"]
    X_te = data["test"]["X"]
    y_te = data["test"]["y"]

    random_state = 1
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=random_state
    )
    ebm1 = ExplainableBoostingClassifier(
        random_state=random_state,
        interactions=0,
    )
    ebm1.fit(X_train, y_train)

    random_state += 10
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.40, random_state=random_state
    )
    ebm2 = ExplainableBoostingClassifier(
        random_state=random_state,
        interactions=0,
    )
    ebm2.fit(X_train, y_train)

    random_state += 10
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.60, random_state=random_state
    )
    ebm3 = ExplainableBoostingClassifier(
        random_state=random_state,
        interactions=0,
    )
    ebm3.fit(X_train, y_train)

    merged_ebm1 = merge_ebms([ebm1, ebm2, ebm3])
    valid_ebm(merged_ebm1)
    global_exp = merged_ebm1.explain_global()
    local_exp = merged_ebm1.explain_local(X_te, y_te)
    smoke_test_explanations(global_exp, local_exp, 6000)

    random_state += 10
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.10, random_state=random_state
    )
    ebm4 = ExplainableBoostingClassifier(random_state=random_state, interactions=0)
    ebm4.fit(X_train, y_train)

    merged_ebm2 = merge_ebms([merged_ebm1, ebm4])
    valid_ebm(merged_ebm2)
    global_exp = merged_ebm2.explain_global()
    local_exp = merged_ebm2.explain_local(X_te, y_te)
    smoke_test_explanations(global_exp, local_exp, 6000)

    random_state += 10
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.50, random_state=random_state
    )
    ebm5 = ExplainableBoostingClassifier(random_state=random_state, interactions=0)
    ebm5.fit(X_train, y_train)

    merged_ebm3 = merge_ebms([ebm5, merged_ebm2])
    valid_ebm(merged_ebm3)
    global_exp = merged_ebm3.explain_global()
    local_exp = merged_ebm3.explain_local(X_te, y_te)
    smoke_test_explanations(global_exp, local_exp, 6000)
