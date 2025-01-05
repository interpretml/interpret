# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import warnings
from functools import partial

import numpy as np
import pytest
from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
    merge_ebms,
)
from interpret.utils import make_synthetic
from sklearn.model_selection import train_test_split

from ...tutils import (
    iris_classification,
    smoke_test_explanations,
)

# arguments for faster fitting time to reduce test time
# we want to test the interface, not get good results
_fast_kwds = {
    "outer_bags": 2,
    "max_rounds": 100,
}


def valid_ebm(ebm):
    assert repr(ebm), "Cannot represent EBM which is important for debugging"
    assert ebm.term_features_[0] == (0,)

    for term_scores in ebm.term_scores_:
        all_finite = np.isfinite(term_scores).all()
        assert all_finite


def test_merge_ebms():
    X, y, names, _ = make_synthetic(classes=2, missing=True, output_type="str")

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

        ebm1 = ExplainableBoostingClassifier(
            names,
            random_state=random_state,
            max_bins=10,
            max_interaction_bins=5,
            interactions=[(8, 3, 0)],
            **_fast_kwds,
        )
        ebm1.fit(X_train, y_train)

        random_state += 10
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.40, random_state=random_state
        )
        ebm2 = ExplainableBoostingClassifier(
            names,
            random_state=random_state,
            max_bins=11,
            max_interaction_bins=4,
            interactions=[(8, 2), (7, 3), (1, 2)],
            **_fast_kwds,
        )
        ebm2.fit(X_train, y_train)

        random_state += 10
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.60, random_state=random_state
        )
        ebm3 = ExplainableBoostingClassifier(
            names,
            random_state=random_state,
            max_bins=12,
            max_interaction_bins=3,
            interactions=[(1, 2), (2, 8)],
            **_fast_kwds,
        )
        ebm3.fit(X_train, y_train)

        merged_ebm1 = merge_ebms([ebm1, ebm2, ebm3])
        valid_ebm(merged_ebm1)
        global_exp = merged_ebm1.explain_global()
        local_exp = merged_ebm1.explain_local(X[:5, :], y[:5])
        smoke_test_explanations(global_exp, local_exp, 6000)

        random_state += 10
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.10, random_state=random_state
        )
        ebm4 = ExplainableBoostingClassifier(
            names,
            random_state=random_state,
            max_bins=13,
            max_interaction_bins=8,
            interactions=2,
            **_fast_kwds,
        )
        ebm4.fit(X_train, y_train)

        merged_ebm2 = merge_ebms([merged_ebm1, ebm4])
        valid_ebm(merged_ebm2)
        global_exp = merged_ebm2.explain_global()
        local_exp = merged_ebm2.explain_local(X[:5, :], y[:5])
        smoke_test_explanations(global_exp, local_exp, 6000)

        random_state += 10
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.50, random_state=random_state
        )
        ebm5 = ExplainableBoostingClassifier(
            names,
            random_state=random_state,
            max_bins=14,
            max_interaction_bins=8,
            interactions=2,
            **_fast_kwds,
        )
        ebm5.fit(X_train, y_train)

        merged_ebm3 = merge_ebms([ebm5, merged_ebm2])
        valid_ebm(merged_ebm3)
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
        max_bins=10,
        **_fast_kwds,
    )
    ebm1.fit(X_train, y_train)

    random_state += 10
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.40, random_state=random_state
    )
    ebm2 = ExplainableBoostingClassifier(
        random_state=random_state,
        interactions=0,
        max_bins=11,
        **_fast_kwds,
    )
    ebm2.fit(X_train, y_train)

    random_state += 10
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.60, random_state=random_state
    )
    ebm3 = ExplainableBoostingClassifier(
        random_state=random_state,
        interactions=0,
        max_bins=12,
        **_fast_kwds,
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
    ebm4 = ExplainableBoostingClassifier(
        random_state=random_state,
        interactions=0,
        max_bins=13,
        # **_fast_kwds,
    )
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
    ebm5 = ExplainableBoostingClassifier(
        random_state=random_state,
        interactions=0,
        max_bins=14,
        # **_fast_kwds,
    )
    ebm5.fit(X_train, y_train)

    merged_ebm3 = merge_ebms([ebm5, merged_ebm2])
    valid_ebm(merged_ebm3)
    global_exp = merged_ebm3.explain_global()
    local_exp = merged_ebm3.explain_local(X_te, y_te)
    smoke_test_explanations(global_exp, local_exp, 6000)


def test_unfitted():
    """To merge EBMs, all have to be fitted."""
    X, y, names, _ = make_synthetic(classes=2, missing=True, output_type="str")
    TestEBM = partial(
        ExplainableBoostingClassifier,
        feature_names=names,
        random_state=42,
        **_fast_kwds,
    )
    ebm1 = TestEBM()
    ebm1.fit(X, y)
    ebm2 = TestEBM()
    # ebm2 is not fitted
    with pytest.raises(Exception, match="All models must be fitted."):
        merge_ebms([ebm1, ebm2])


def test_merge_monotone():
    """Check merging of features with `monotone_constraints`."""
    X, y, names, _ = make_synthetic(classes=None, missing=True, output_type="str")
    TestEBM = partial(
        ExplainableBoostingRegressor,
        feature_names=names,
        random_state=42,
        **_fast_kwds,
    )
    # feature 3, 6 are truly monotonous increasing, 7 has no impact
    ebm1 = TestEBM(monotone_constraints=[0, 0, 0, +1, 0, 0, +1, +1, 0, 0])
    ebm1.fit(X, y)
    ebm2 = TestEBM(monotone_constraints=[0, 0, 0, +1, 0, 0, +0, -1, 0, 0])
    ebm2.fit(X, y)
    merged_ebm = merge_ebms([ebm1, ebm2])
    assert merged_ebm.monotone_constraints == [0, 0, 0, +1, 0, 0, +0, +0, 0, 0]
    merged_ebm = merge_ebms([ebm2, ebm2])
    assert merged_ebm.monotone_constraints == [0, 0, 0, +1, 0, 0, +0, -1, 0, 0]
    ebm3 = TestEBM(monotone_constraints=None)
    ebm3.fit(X, y)
    merged_ebm = merge_ebms([ebm1, ebm2, ebm3])
    assert merged_ebm.monotone_constraints is None


def test_merge_exclude():
    """Check merging of features with `exclude`."""
    X, y, names, _ = make_synthetic(classes=2, missing=True, output_type="str")
    TestEBM = partial(
        ExplainableBoostingClassifier,
        feature_names=names,
        random_state=42,
        **_fast_kwds,
    )
    ebm1 = TestEBM(exclude=None)
    ebm1.fit(X, y)
    ebm2 = TestEBM(exclude=[0, 1, 2])
    ebm2.fit(X, y)
    merged_ebm = merge_ebms([ebm1, ebm2])
    assert merged_ebm.exclude is None
    ebm1 = TestEBM(exclude=[0, 2])
    ebm1.fit(X, y)
    ebm2 = TestEBM(exclude=[0, 1, 2])
    ebm2.fit(X, y)
    merged_ebm = merge_ebms([ebm1, ebm2])
    assert merged_ebm.exclude == [(0,), (2,)]
    ebm1 = TestEBM(exclude="mains")
    ebm1.fit(X, y)
    ebm2 = TestEBM(exclude=[0, 1, 2])
    ebm2.fit(X, y)
    merged_ebm = merge_ebms([ebm1, ebm2])
    assert merged_ebm.exclude == [(0,), (1,), (2,)]
