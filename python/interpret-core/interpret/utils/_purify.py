# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license
# Author: Paul Koch <code@koch.ninja>

# For more details, please refer to the paper:
# https://arxiv.org/abs/1911.04974

import numpy as np

from ._native import Native


def _measure_impurity(scores, weights):
    native = Native.get_native_singleton()
    return native.measure_impurity(scores, weights).sum()


def purify(scores, weights, tolerance=0.0, is_randomized=True):
    """Purifies a score tensor into it's pure component and a series of impure components. For pairs, the
        result will be a pair where the weighted sum along any row or column is zero, and the two main effects
        which are the impurities from the pair. The main effects will be further purified into zero-centered graphs
        and an intercept. This function also handles multiclass, which is detected when the scores tensor has one
        additional dimension than the weights. For multiclass, the class scores are adjusted to sum to 0,
        which can be done without generating any impurities.

        Purification algorithm is based on the paper:
        "Purifying Interaction Effects with the Functional ANOVA: An Efficient Algorithm for Recovering Identifiable Additive Models"
        https://arxiv.org/abs/1911.04974

    Args:
        scores: The score tensor to be purified.
        weights: Weights for each element in the scores tensor.
        tolerance: If needed, a tolerance can be specified to make the algorithm exit faster at the expense of purity.
            The algorithm will exit whenever either the tolerance is reached, or when purity stops improving.
        is_randomized: If is_randomized is False, then purification happens in a predictable order. Using random
            ordering removes the bias that might be introduced by always processing one dimension first.

    Returns:
        A tuple containing 3 values:
            1) The purified tensor.
            2) A list of tuples of the impurities generated. The first item in the tuple will
               be a tuple of indexes which indicates which dimensions the impurity applies to. The second
               item is the impurity tensor for those dimensions.
            3) The intercept generated from purification.
    """

    if scores.ndim != weights.ndim and scores.ndim != weights.ndim + 1:
        msg = "scores and weights do not match in terms of dimensionality."
        raise Exception(msg)

    n_dim = weights.ndim
    if n_dim == 0:
        msg = "scores cannot have zero dimensions."
        raise Exception(msg)

    scores = scores.copy()
    native = Native.get_native_singleton()
    impurities = []
    n_possible = (1 << n_dim) - 1
    prev_level = [None] * n_possible
    prev_level[0] = [scores, weights]
    next_level = [None] * n_possible
    intercept = np.zeros(
        scores.shape[-1] if scores.ndim != weights.ndim else 1, np.float64
    )
    for n_dimensions in range(n_dim, 1, -1):
        for dims in range(n_possible):
            items = prev_level[dims]
            if items is None:
                continue
            level_scores, level_weights = items
            prev_level[dims] = None

            level_impurities, level_intercept = native.purify(
                level_scores, level_weights, tolerance, is_randomized
            )
            intercept += level_intercept
            if dims != 0:
                # do not insert the original score tensor into the impurities
                key = tuple(
                    n_dim - 1 - i
                    for i in range(n_dim - 1, -1, -1)
                    if ((1 << i) & dims) == 0
                )
                impurities.append((key, level_scores))

            impure_idx = 0
            for dim_idx in range(n_dim):
                if (1 << dim_idx) & dims != 0:
                    continue
                new_dims = dims | (1 << dim_idx)
                items = next_level[new_dims]
                if items is not None:
                    items[0] += level_impurities[impure_idx]
                else:
                    next_level[new_dims] = [
                        level_impurities[impure_idx],
                        level_weights.sum(axis=n_dimensions - 1 - impure_idx),
                    ]
                impure_idx += 1
        temp = next_level
        next_level = prev_level
        prev_level = temp

    for dims in range(n_possible):
        items = prev_level[dims]
        if items is None:
            continue
        level_scores, level_weights = items

        _, level_intercept = native.purify(
            level_scores, level_weights, tolerance, is_randomized
        )
        intercept += level_intercept

        if dims != 0:
            key = tuple(
                n_dim - 1 - i
                for i in range(n_dim - 1, -1, -1)
                if ((1 << i) & dims) == 0
            )
            impurities.append((key, level_scores))

    return scores, impurities, intercept
