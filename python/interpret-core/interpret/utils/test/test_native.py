# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from .._native import Native, Booster

import numpy as np
import ctypes as ct
from contextlib import closing

from scipy.stats import normaltest, shapiro

import pytest


def test_hist():
    np.random.seed(0)
    X_col = np.random.random_sample((1000,))
    counts, vals = np.histogram(X_col, bins="doane")

    X_col = np.concatenate(([np.nan], X_col))
    
    native = Native.get_native_singleton()
    n_cuts = native.get_histogram_cut_count(X_col)

    cuts = native.cut_uniform(X_col, n_cuts)
    bin_indexes = native.discretize(X_col, cuts)
    bin_counts = np.bincount(bin_indexes, minlength=len(cuts) + 2)
    edges = np.concatenate(([np.nanmin(X_col)], cuts, [np.nanmax(X_col)]))

    assert bin_counts[0] == 1
    assert(np.sum(bin_counts) == 1000 + 1)
    bin_counts = bin_counts[1:]

    assert np.array_equal(counts, bin_counts)
    assert np.allclose(vals, edges)

def test_cut_winsorized():
    np.random.seed(0)
    X_col = np.arange(-10, 90)
    X_col = np.concatenate(([np.nan], [-np.inf], [-np.inf], X_col, [np.inf], [np.inf], [np.inf]))
    
    native = Native.get_native_singleton()

    cuts = native.cut_winsorized(X_col, 10)
    bin_indexes = native.discretize(X_col, cuts)
    bin_counts = np.bincount(bin_indexes, minlength=len(cuts) + 2)

    assert len(cuts) == 10
    assert(np.sum(bin_counts) == 106)
    assert bin_counts[0] == 1

def test_suggest_graph_bound():
    native = Native.get_native_singleton()
    cuts=[25, 50, 75]
    (low_graph_bound, high_graph_bound) = native.suggest_graph_bounds(cuts, 24, 76)
    assert low_graph_bound == 24
    assert high_graph_bound == 76

def test_suggest_graph_bound_no_min_max():
    native = Native.get_native_singleton()
    cuts=[25, 50, 75]
    (low_graph_bound, high_graph_bound) = native.suggest_graph_bounds(cuts)
    assert low_graph_bound < 25
    assert 75 < high_graph_bound

def test_suggest_graph_bound_no_cuts():
    native = Native.get_native_singleton()
    cuts=[]
    (low_graph_bound, high_graph_bound) = native.suggest_graph_bounds(cuts, 24, 76)
    assert low_graph_bound == 24
    assert high_graph_bound == 76

def test_gaussian_random_number_generator():
    # Tests normality of the gaussian RNG with shapiro and D'Agostino/Pearson's tests. 
    # Caution: can fail with extremely low probability. TODO: Harsha calculate this failure prob.

    stddevs = [1, 2]
    n_iter = 1000

    native = Native.get_native_singleton()

    for std in stddevs:
        norm_results, shapiro_results = [], []
        for i in range(n_iter):
            rands = native.generate_gaussian_random(None, std, count=1000)
            norm_results.append(normaltest(rands).pvalue > 0.05)
            shapiro_results.append(shapiro(rands).pvalue > 0.05)

        assert 0.9 < np.mean(norm_results) < 0.99
        assert 0.9 < np.mean(shapiro_results) < 0.99