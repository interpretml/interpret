# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license
# Author: Paul Koch <code@koch.ninja>

import numpy as np
import pytest

from interpret.utils import purify
from interpret.utils._purify import _measure_impurity


def test_purify_regression_1():
    shape = (255)

    scores = np.random.uniform(low=-0.5, high=1.0, size=shape) 
    weights = np.random.uniform(low=1.0, high=10.0, size=shape) 
    purified, impurities, intercept = purify(scores, weights)

    refilled = purified + intercept

    assert(_measure_impurity(purified, weights) < 0.2)
    assert(len(impurities) == 0)
    assert np.allclose(refilled, scores)


def test_purify_multiclass_1():
    shape = (257, 9)

    scores = np.random.uniform(low=-0.5, high=1.0, size=shape)
    weights = np.random.uniform(low=1.0, high=10.0, size=shape[:-1])
    purified, impurities, intercept = purify(scores, weights)

    refilled = purified + intercept

    assert(_measure_impurity(purified, weights) < 1.5)
    assert(len(impurities) == 0)
    assert np.allclose(refilled, scores)


def test_purify_regression_2():
    shape = (31, 33)

    scores = np.random.uniform(low=-0.5, high=1.0, size=shape) 
    weights = np.random.uniform(low=1.0, high=10.0, size=shape) 
    purified, impurities, intercept = purify(scores, weights)

    refilled = purified + intercept

    assert(_measure_impurity(purified, weights) < 0.2)
    
    for idxes, impurity in impurities:
        axis = tuple(i for i in range(weights.ndim) if i not in idxes)
        collapsed_weight = weights.sum(axis=axis)
        assert(_measure_impurity(impurity, collapsed_weight) < 0.2)
        new_shape = [1] * len(shape)
        for i in idxes:
            new_shape[i] = shape[i]
        refilled += impurity.reshape(new_shape)

    assert np.allclose(refilled, scores)


def test_purify_multiclass_2():
    shape = (33, 31, 5)

    scores = np.random.uniform(low=-0.5, high=1.0, size=shape)
    weights = np.random.uniform(low=1.0, high=10.0, size=shape[:-1])
    purified, impurities, intercept = purify(scores, weights)

    refilled = purified + intercept

    assert(_measure_impurity(purified, weights) < 1.5)
    
    for idxes, impurity in impurities:
        axis = tuple(i for i in range(weights.ndim) if i not in idxes)
        collapsed_weight = weights.sum(axis=axis)
        assert(_measure_impurity(impurity, collapsed_weight) < 1.5)
        new_shape = [1] * len(shape)
        new_shape[-1] = shape[-1]  # keep the same # of classes
        for i in idxes:
            new_shape[i] = shape[i]
        refilled += impurity.reshape(new_shape)

    assert np.allclose(refilled, scores)


def test_purify_regression_3():
    shape = (4, 5, 6)

    scores = np.random.uniform(low=-0.5, high=1.0, size=shape) 
    weights = np.random.uniform(low=1.0, high=10.0, size=shape) 
    purified, impurities, intercept = purify(scores, weights)

    refilled = purified + intercept

    assert(_measure_impurity(purified, weights) < 0.2)
    
    for idxes, impurity in impurities:
        axis = tuple(i for i in range(weights.ndim) if i not in idxes)
        collapsed_weight = weights.sum(axis=axis)
        assert(_measure_impurity(impurity, collapsed_weight) < 0.2)
        new_shape = [1] * len(shape)
        for i in idxes:
            new_shape[i] = shape[i]
        refilled += impurity.reshape(new_shape)

    assert np.allclose(refilled, scores)


def test_purify_multiclass_3():
    shape = (4, 5, 6, 4)

    scores = np.random.uniform(low=-0.5, high=1.0, size=shape)
    weights = np.random.uniform(low=1.0, high=10.0, size=shape[:-1])
    purified, impurities, intercept = purify(scores, weights)

    refilled = purified + intercept

    assert(_measure_impurity(purified, weights) < 1.5)
    
    for idxes, impurity in impurities:
        axis = tuple(i for i in range(weights.ndim) if i not in idxes)
        collapsed_weight = weights.sum(axis=axis)
        assert(_measure_impurity(impurity, collapsed_weight) < 1.5)
        new_shape = [1] * len(shape)
        new_shape[-1] = shape[-1]  # keep the same # of classes
        for i in idxes:
            new_shape[i] = shape[i]
        refilled += impurity.reshape(new_shape)

    assert np.allclose(refilled, scores)


def test_purify_regression_5():
    shape = (4, 5, 6, 3, 7)

    scores = np.random.uniform(low=-0.5, high=1.0, size=shape) 
    weights = np.random.uniform(low=1.0, high=10.0, size=shape) 
    purified, impurities, intercept = purify(scores, weights)

    refilled = purified + intercept

    assert(_measure_impurity(purified, weights) < 0.2)
    
    for idxes, impurity in impurities:
        axis = tuple(i for i in range(weights.ndim) if i not in idxes)
        collapsed_weight = weights.sum(axis=axis)
        assert(_measure_impurity(impurity, collapsed_weight) < 0.2)
        new_shape = [1] * len(shape)
        for i in idxes:
            new_shape[i] = shape[i]
        refilled += impurity.reshape(new_shape)

    assert np.allclose(refilled, scores)


def test_purify_multiclass_5():
    shape = (4, 5, 6, 3, 7, 3)

    scores = np.random.uniform(low=-0.5, high=1.0, size=shape)
    weights = np.random.uniform(low=1.0, high=10.0, size=shape[:-1])
    purified, impurities, intercept = purify(scores, weights)

    refilled = purified + intercept

    assert(_measure_impurity(purified, weights) < 1.5)
    
    for idxes, impurity in impurities:
        axis = tuple(i for i in range(weights.ndim) if i not in idxes)
        collapsed_weight = weights.sum(axis=axis)
        assert(_measure_impurity(impurity, collapsed_weight) < 1.5)
        new_shape = [1] * len(shape)
        new_shape[-1] = shape[-1]  # keep the same # of classes
        for i in idxes:
            new_shape[i] = shape[i]
        refilled += impurity.reshape(new_shape)

    assert np.allclose(refilled, scores)
