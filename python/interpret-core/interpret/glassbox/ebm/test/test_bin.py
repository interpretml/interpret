# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import pytest
import numpy as np
import math

from ..bin import eval_terms, make_bin_weights, ebm_decision_function
from ....utils._binning import clean_X

def test_eval_terms():
    X = np.array([["a", 1, np.nan], ["b", 2, 8], ["a", 2, 9], [None, 3, "BAD_CONTINUOUS"]], dtype=np.object_)
    feature_names_in = ["f1", "99", "f3"]
    feature_types_in = ['nominal', 'nominal', 'continuous']

    shared_categores = {"a": 1} # "b" is unknown category
    shared_cuts = np.array([8.5], dtype=np.float64)

    # for level 1, "b" is unknown category
    # for level 1, we combine "2" and "3" into one category!
    # for level 2, collapse all our categories to keep the tensor small for testing
    bins = [
        [{"a": 1, "b": 2}, shared_categores, shared_categores],
        [{"1": 1, "2": 2, "3": 3}, {"1": 1, "2": 2, "3": 2}, {"1": 1, "2": 1, "3": 1}],
        [shared_cuts, shared_cuts, np.array([], dtype=np.float64)]
    ]

    term_features = []
    term_scores = []

    term_features.append([0])
    term_scores.append(np.array([0.1, 0.2, 0.3, 0], dtype=np.float64))

    term_features.append([1])
    term_scores.append(np.array([0.01, 0.02, 0.03, 0.04, 0], dtype=np.float64))

    term_features.append([2])
    term_scores.append(np.array([0.001, 0.002, 0.003, 0], dtype=np.float64))

    term_features.append([0, 1])
    term_scores.append(np.array([[0.0001, 0.0002, 0.0003, 0], [0.0004, 0.0005, 0.0006, 0], [0, 0, 0, 0]], dtype=np.float64))

    term_features.append([0, 2])
    term_scores.append(np.array([[0.00001, 0.00002, 0.00003, 0], [0.00004, 0.00005, 0.00006, 0], [0, 0, 0, 0]], dtype=np.float64))

    term_features.append([0, 1, 2])
    term_scores.append(np.array([[[0.000001, 0.000002, 0], [0.000003, 0.000004, 0], [0, 0, 0]], [[0.000005, 0.000006, 0], [0.000007, 0.000008, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=np.float64))

    X, n_samples = clean_X(X)

    bin_weights = make_bin_weights(X, n_samples, None, feature_names_in, feature_types_in, bins, term_features)
    assert(bin_weights is not None)

    result = list(eval_terms(X, n_samples, feature_names_in, feature_types_in, bins, term_features))
    result = [term_scores[x[0]][tuple(x[1])] for x in result]

    assert(result[0][0] == 0.2)
    assert(result[0][1] == 0.3)
    assert(result[0][2] == 0.2)
    assert(result[0][3] == 0.1)

    assert(result[1][0] == 0.02)
    assert(result[1][1] == 0.03)
    assert(result[1][2] == 0.03)
    assert(result[1][3] == 0.04)

    assert(result[2][0] == 0.001)
    assert(result[2][1] == 0.002)
    assert(result[2][2] == 0.003)
    assert(result[2][3] == 0)

    # term4 finishes before term3 since shared_cuts allows the 3rd feature to be completed first
    assert(result[4][0] == 0.0005)
    assert(result[4][1] == 0)
    assert(result[4][2] == 0.0006)
    assert(result[4][3] == 0.0003)

    # term4 finishes before term3 since shared_cuts allows the 3rd feature to be completed first
    assert(result[3][0] == 0.00004)
    assert(result[3][1] == 0)
    assert(result[3][2] == 0.00006)
    assert(result[3][3] == 0)

    assert(result[5][0] == 0.000007)
    assert(result[5][1] == 0)
    assert(result[5][2] == 0.000008)
    assert(result[5][3] == 0)

    scores = ebm_decision_function(X, n_samples, feature_names_in, feature_types_in, bins, np.array([7], dtype=np.float64), term_scores, term_features)
    assert(math.isclose(scores[0], 7.221547))
    assert(math.isclose(scores[1], 7.332000))
    assert(math.isclose(scores[2], 7.233668))
    assert(math.isclose(scores[3], 7.140300))
