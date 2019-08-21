# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..postprocessing import multiclass_postprocess
import numpy as np


def test_multiclass_postprocess_smoke():
    n = 1000
    d = 2
    k = 3
    b = 10

    X_binned = np.random.randint(b, size=(n, d))
    feature_graphs = []
    for _ in range(d):
        feature_graphs.append(np.random.rand(b, k))

    def binned_predict_proba(X_binned, k=3):
        n = len(X_binned)
        return 1 / k * np.ones((n, k))

    feature_types = ['numeric'] * d
    results = multiclass_postprocess(X_binned, feature_graphs, binned_predict_proba, feature_types)

    assert "intercept" in results
    assert "feature_graphs" in results
