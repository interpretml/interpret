# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license
# Author: Paul Koch <code@koch.ninja>

# For more details, please refer to the paper:
# https://arxiv.org/abs/1911.04974

import numpy as np

from ._native import Native


def _measure_impurity(scores, weights):
    native = Native.get_native_singleton()
    impurity = native.measure_impurity(scores, weights).sum()
    return impurity


def purify(scores, weights, tolerance=0.0, is_randomized=True):
    if scores.ndim != weights.ndim and scores.ndim != weights.ndim + 1:
        raise Exception("scores and weights do not match in terms of dimensionality.")

    n_dim = weights.ndim
    if n_dim == 0:
        raise Exception("scores cannot have zero dimensions.")

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


# TODO: Apply purification to EBMs either (based on a boolean option that we can expose publicly):
#    1) After all boosting is complete.  We can either throw away the lower dimensional contributions,
#       or move the score contributions to the lower dimensional terms based on benchmarking results.
#    2) During boosting, where we would throw away the impure components so that
#       the algorithm would not overfit the lower dimensional components.
#       - This would be especially important when we boost mains and interactions together at
#         the same time because we don't want the model to force feed some mains that just happen
#         to be included in an interaction.
