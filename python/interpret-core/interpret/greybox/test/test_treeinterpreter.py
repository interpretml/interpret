# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
from ..treeinterpreter import TreeInterpreter

import pytest


# TODO: Stop ignoring when treeinterpreter updates upstream.
@pytest.mark.skip
def test_that_tree_works():
    from treeinterpreter import treeinterpreter as ti
    # Code below compares refactored blog post to our wrapper implementation.
    # http://blog.datadive.net/random-forest-interpretation-with-scikit-learn/

    # Fit tree
    boston = load_boston()
    rf = RandomForestRegressor()
    X, y = boston.data[:300], boston.target[:300]
    feature_names = boston.feature_names

    X_new = boston.data[[300, 309]]
    y_new = boston.target[[300, 309]]
    rf.fit(X, y)

    # Build expected local explanation
    prediction, bias, contributions = ti.predict(rf, X_new)

    # Build actual local explanation
    explainer = TreeInterpreter(rf, X, feature_names=feature_names)
    local_expl = explainer.explain_local(X_new, y_new)

    a_local_data = local_expl.data(key=0)
    assert all(
        [
            feature_names[i] == a_local_data["names"][i]
            for i in range(len(feature_names))
        ]
    )
    assert all(
        [
            contributions[0, i] == a_local_data["scores"][i]
            for i in range(len(feature_names))
        ]
    )
    assert a_local_data["extra"]["names"][0] == "Bias"
    assert a_local_data["extra"]["scores"][0] == bias[0]
