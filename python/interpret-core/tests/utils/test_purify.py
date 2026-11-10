# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license
# Author: Paul Koch <code@koch.ninja>

import itertools

import numpy as np
from interpret.utils import purify
from interpret.utils._purify import _measure_impurity


def test_purify_regression_0():
    shape = (5, 0, 5)

    keys = set(
        itertools.chain.from_iterable(
            itertools.combinations(range(len(shape)), n) for n in range(1, len(shape))
        )
    )

    scores = np.empty(shape, float)
    weights = np.empty(shape, float)
    purified, impurities, intercept = purify(scores, weights)

    assert intercept == 0.0
    assert purified.shape == shape
    assert len(keys) == len(impurities)

    for idxes, impurity in impurities:
        assert idxes in keys
        assert np.all(impurity == 0.0)


def test_purify_regression_1():
    shape = (255,)

    scores = np.random.uniform(low=-0.5, high=1.0, size=shape)
    weights = np.random.uniform(low=1.0, high=10.0, size=shape)
    purified, impurities, intercept = purify(scores, weights)

    assert _measure_impurity(purified, weights) < 0.2

    refilled = purified + intercept

    assert len(impurities) == 0
    assert np.allclose(refilled, scores)


def test_purify_multiclass_1():
    shape = (257,)
    n_classes = 9

    scores = np.random.uniform(low=-0.5, high=1.0, size=(*shape, n_classes))
    weights = np.random.uniform(low=1.0, high=10.0, size=shape)
    purified, impurities, intercept = purify(scores, weights)

    assert _measure_impurity(purified, weights) < 1.5

    refilled = purified + intercept

    # make them identifiable for checking
    refilled -= np.average(refilled, axis=-1, keepdims=True)
    scores -= np.average(scores, axis=-1, keepdims=True)

    assert len(impurities) == 0
    assert np.allclose(refilled, scores)


def test_purify_regression_2():
    shape = (31, 33)

    keys = set(
        itertools.chain.from_iterable(
            itertools.combinations(range(len(shape)), n) for n in range(1, len(shape))
        )
    )

    scores = np.random.uniform(low=-0.5, high=1.0, size=shape)
    weights = np.random.uniform(low=1.0, high=10.0, size=shape)
    purified, impurities, intercept = purify(scores, weights)

    assert len(keys) == len(impurities)
    assert _measure_impurity(purified, weights) < 0.2

    refilled = purified + intercept

    for idxes, impurity in impurities:
        assert idxes in keys
        axis = tuple(i for i in range(weights.ndim) if i not in idxes)
        collapsed_weight = weights.sum(axis=axis)
        assert _measure_impurity(impurity, collapsed_weight) < 0.2
        new_shape = [1] * len(shape)
        for i in idxes:
            new_shape[i] = shape[i]
        refilled += impurity.reshape(new_shape)

    assert np.allclose(refilled, scores)


def test_purify_multiclass_2():
    shape = (33, 31)
    n_classes = 5

    keys = set(
        itertools.chain.from_iterable(
            itertools.combinations(range(len(shape)), n) for n in range(1, len(shape))
        )
    )

    scores = np.random.uniform(low=-0.5, high=1.0, size=(*shape, n_classes))
    weights = np.random.uniform(low=1.0, high=10.0, size=shape)
    purified, impurities, intercept = purify(scores, weights)

    assert len(keys) == len(impurities)
    assert _measure_impurity(purified, weights) < 1.5

    refilled = purified + intercept

    for idxes, impurity in impurities:
        assert idxes in keys
        axis = tuple(i for i in range(weights.ndim) if i not in idxes)
        collapsed_weight = weights.sum(axis=axis)
        assert _measure_impurity(impurity, collapsed_weight) < 1.5
        new_shape = [1] * (len(shape) + 1)
        new_shape[-1] = n_classes  # keep the same # of classes
        for i in idxes:
            new_shape[i] = shape[i]
        refilled += impurity.reshape(new_shape)

    # make them identifiable for checking
    refilled -= np.average(refilled, axis=-1, keepdims=True)
    scores -= np.average(scores, axis=-1, keepdims=True)

    assert np.allclose(refilled, scores)


def test_purify_regression_3():
    shape = (4, 5, 6)

    keys = set(
        itertools.chain.from_iterable(
            itertools.combinations(range(len(shape)), n) for n in range(1, len(shape))
        )
    )

    scores = np.random.uniform(low=-0.5, high=1.0, size=shape)
    weights = np.random.uniform(low=1.0, high=10.0, size=shape)
    purified, impurities, intercept = purify(scores, weights)

    assert len(keys) == len(impurities)
    assert _measure_impurity(purified, weights) < 0.2

    refilled = purified + intercept

    for idxes, impurity in impurities:
        assert idxes in keys
        axis = tuple(i for i in range(weights.ndim) if i not in idxes)
        collapsed_weight = weights.sum(axis=axis)
        assert _measure_impurity(impurity, collapsed_weight) < 0.2
        new_shape = [1] * len(shape)
        for i in idxes:
            new_shape[i] = shape[i]
        refilled += impurity.reshape(new_shape)

    assert np.allclose(refilled, scores)


def test_purify_multiclass_3():
    shape = (4, 5, 6)
    n_classes = 4

    keys = set(
        itertools.chain.from_iterable(
            itertools.combinations(range(len(shape)), n) for n in range(1, len(shape))
        )
    )

    scores = np.random.uniform(low=-0.5, high=1.0, size=(*shape, n_classes))
    weights = np.random.uniform(low=1.0, high=10.0, size=shape)
    purified, impurities, intercept = purify(scores, weights)

    assert len(keys) == len(impurities)
    assert _measure_impurity(purified, weights) < 1.5

    refilled = purified + intercept

    for idxes, impurity in impurities:
        assert idxes in keys
        axis = tuple(i for i in range(weights.ndim) if i not in idxes)
        collapsed_weight = weights.sum(axis=axis)
        assert _measure_impurity(impurity, collapsed_weight) < 1.5
        new_shape = [1] * (len(shape) + 1)
        new_shape[-1] = n_classes  # keep the same # of classes
        for i in idxes:
            new_shape[i] = shape[i]
        refilled += impurity.reshape(new_shape)

    # make them identifiable for checking
    refilled -= np.average(refilled, axis=-1, keepdims=True)
    scores -= np.average(scores, axis=-1, keepdims=True)

    assert np.allclose(refilled, scores)


def test_purify_regression_5():
    shape = (4, 5, 6, 3, 7)

    keys = set(
        itertools.chain.from_iterable(
            itertools.combinations(range(len(shape)), n) for n in range(1, len(shape))
        )
    )

    scores = np.random.uniform(low=-0.5, high=1.0, size=shape)
    weights = np.random.uniform(low=1.0, high=10.0, size=shape)
    purified, impurities, intercept = purify(scores, weights)

    assert len(keys) == len(impurities)
    assert _measure_impurity(purified, weights) < 0.2

    refilled = purified + intercept

    for idxes, impurity in impurities:
        assert idxes in keys
        axis = tuple(i for i in range(weights.ndim) if i not in idxes)
        collapsed_weight = weights.sum(axis=axis)
        assert _measure_impurity(impurity, collapsed_weight) < 0.2
        new_shape = [1] * len(shape)
        for i in idxes:
            new_shape[i] = shape[i]
        refilled += impurity.reshape(new_shape)

    assert np.allclose(refilled, scores)


def test_purify_multiclass_5():
    shape = (4, 5, 6, 3, 7)
    n_classes = 3

    keys = set(
        itertools.chain.from_iterable(
            itertools.combinations(range(len(shape)), n) for n in range(1, len(shape))
        )
    )

    scores = np.random.uniform(low=-0.5, high=1.0, size=(*shape, n_classes))
    weights = np.random.uniform(low=1.0, high=10.0, size=shape)
    purified, impurities, intercept = purify(scores, weights)

    assert len(keys) == len(impurities)
    assert _measure_impurity(purified, weights) < 1.5

    refilled = purified + intercept

    for idxes, impurity in impurities:
        assert idxes in keys
        axis = tuple(i for i in range(weights.ndim) if i not in idxes)
        collapsed_weight = weights.sum(axis=axis)
        assert _measure_impurity(impurity, collapsed_weight) < 1.5
        new_shape = [1] * (len(shape) + 1)
        new_shape[-1] = n_classes  # keep the same # of classes
        for i in idxes:
            new_shape[i] = shape[i]
        refilled += impurity.reshape(new_shape)

    # make them identifiable for checking
    refilled -= np.average(refilled, axis=-1, keepdims=True)
    scores -= np.average(scores, axis=-1, keepdims=True)

    assert np.allclose(refilled, scores)


def _random_empty_cell_inputs(rng, shape, p_empty, n_classes=None):
    weights = rng.uniform(0.1, 3.0, shape) * (rng.random(shape) > p_empty)
    score_shape = shape if n_classes is None else (*shape, n_classes)
    scores = rng.normal(size=score_shape)
    return scores, weights


def test_purify_randomized_with_empty_cells_regression():
    # A fresh random visiting order on each sweep used to end the purification at the first
    # sweep that did not shrink the total impurity, which left such tensors impure.
    rng = np.random.default_rng(0)
    for shape, p_empty in (((5, 6), 0.6), ((3, 3, 3), 0.3), ((4, 5, 6), 0.3)):
        for _ in range(10):
            scores, weights = _random_empty_cell_inputs(rng, shape, p_empty)
            purified, impurities, intercept = purify(
                scores, weights, tolerance=0.0, is_randomized=True
            )

            for axis in range(len(shape)):
                assert np.allclose((purified * weights).sum(axis=axis), 0.0, atol=1e-8)

            refilled = purified + intercept
            for idxes, impurity in impurities:
                new_shape = [1] * len(shape)
                for i in idxes:
                    new_shape[i] = shape[i]
                refilled += impurity.reshape(new_shape)
            assert np.allclose(refilled, scores)


def test_purify_randomized_multiclass_with_empty_cells_regression():
    rng = np.random.default_rng(1)
    shape = (5, 6)
    n_classes = 3
    for _ in range(10):
        scores, weights = _random_empty_cell_inputs(rng, shape, 0.6, n_classes)
        purified, _, _ = purify(scores, weights, tolerance=0.0, is_randomized=True)

        weighted = purified * weights[..., np.newaxis]
        for axis in range(len(shape)):
            assert np.allclose(weighted.sum(axis=axis), 0.0, atol=1e-8)
