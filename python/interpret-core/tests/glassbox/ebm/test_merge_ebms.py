# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from interpret.utils import synthetic_default

from ...tutils import (
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
    # TODO: improve this test by checking the merged ebms for validity. 
    #       Right now the merged ebms fail the check for valid_ebm.
    #       The failure might be related to the warning we're getting
    #       about the scalar divide in the merge_ebms line:
    #       "percentage.append((new_high - new_low) / (old_high - old_low))"
    
    X, y, names, _ = synthetic_default(classes=2, missing=True, objects=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Missing values detected.*")
        warnings.filterwarnings("ignore", "Dropping term.*")
        warnings.filterwarnings(
            "ignore",
            "Interactions with 3 or more terms are not graphed in global explanations.*",
        )

        random_state = 1
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.20, random_state=random_state
        )
        X_train[:, 3] = ""
        X_train[0, 3] = "-1"
        X_train[1, 3] = "-1."
        X_train[2, 3] = "-1.0"
        # make this categorical to merge with the continuous feature in the other ebms
        X_train[3, 3] = "me"
        X_train[4, 3] = "you"

        ebm1 = ExplainableBoostingClassifier(names, 
            random_state=random_state,
            max_interaction_bins=5,
            interactions=[(8, 3, 0)],
        )
        ebm1.fit(X_train, y_train)

        random_state += 10
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.40, random_state=random_state
        )
        ebm2 = ExplainableBoostingClassifier(names, 
            random_state=random_state,
            max_interaction_bins=4,
            interactions=[(8, 2), (7, 3), (1, 2)],
        )
        ebm2.fit(X_train, y_train)

        random_state += 10
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.60, random_state=random_state
        )
        ebm3 = ExplainableBoostingClassifier(names, 
            random_state=random_state,
            max_interaction_bins=3,
            interactions=[(1, 2), (2, 8)],
        )
        ebm3.fit(X_train, y_train)

        merged_ebm1 = merge_ebms([ebm1, ebm2, ebm3])
        #valid_ebm(merged_ebm1)
        global_exp = merged_ebm1.explain_global()
        local_exp = merged_ebm1.explain_local(X[:5, :], y[:5])
        smoke_test_explanations(global_exp, local_exp, 6000)

        random_state += 10
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.10, random_state=random_state
        )
        ebm4 = ExplainableBoostingClassifier(names, 
            random_state=random_state, max_interaction_bins=8, interactions=2
        )
        ebm4.fit(X_train, y_train)

        merged_ebm2 = merge_ebms([merged_ebm1, ebm4])
        #valid_ebm(merged_ebm2)
        global_exp = merged_ebm2.explain_global()
        local_exp = merged_ebm2.explain_local(X[:5, :], y[:5])
        smoke_test_explanations(global_exp, local_exp, 6000)

        random_state += 10
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.50, random_state=random_state
        )
        ebm5 = ExplainableBoostingClassifier(names, 
            random_state=random_state, max_interaction_bins=8, interactions=2
        )
        ebm5.fit(X_train, y_train)

        merged_ebm3 = merge_ebms([ebm5, merged_ebm2])
        #valid_ebm(merged_ebm3)
        global_exp = merged_ebm3.explain_global()
        local_exp = merged_ebm3.explain_local(X[:5, :], y[:5])
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
