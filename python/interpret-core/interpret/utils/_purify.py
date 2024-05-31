# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license
# Author: Paul Koch <code@koch.ninja>

# For more details, please refer to the paper:
# https://arxiv.org/abs/1911.04974

import numpy as np

from ._native import Native


def _measure_impurity(scores, weights):
    if scores.ndim != weights.ndim:
        if scores.ndim != weights.ndim + 1:
            raise Exception(
                "scores and weights do not match in terms of dimensionality."
            )
        # multiclass means the scores have the class scores in the last dimension
        return sum(
            _measure_impurity(scores[..., i], weights) for i in range(scores.shape[-1])
        )

    shape = scores.shape
    exclude_idx = len(shape) - 1
    tensor_index = [0] * len(shape)
    total_system = 0.0
    while True:
        total_equation = 0.0
        for bin_idx in range(shape[exclude_idx]):
            tensor_index[exclude_idx] = bin_idx
            tupple_index = tuple(tensor_index)
            total_equation += weights[tupple_index] * scores[tupple_index]
        tensor_index[exclude_idx] = 0
        total_system += abs(total_equation)

        dim_idx = len(shape) - 1
        while True:
            if dim_idx != exclude_idx:
                bin_idx = tensor_index[dim_idx] + 1
                tensor_index[dim_idx] = bin_idx
                if bin_idx != shape[dim_idx]:
                    break
                tensor_index[dim_idx] = 0
            if dim_idx == 0:
                exclude_idx -= 1
                break
            dim_idx -= 1
        if exclude_idx < 0:
            break
    return total_system


def _purify_downstream(scores, weights, tolerance, is_randomized):
    n_dim = scores.ndim
    if n_dim == 0:
        raise Exception("scores cannot have zero dimensions.")

    scores = scores.copy()
    native = Native.get_native_singleton()
    impurities = []
    n_possible = (1 << n_dim) - 1
    prev_level = [None] * n_possible
    prev_level[0] = [scores, weights]
    next_level = [None] * n_possible
    intercept = 0.0
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
def purify(scores, weights, tolerance=1e-6, is_randomized=True):
    if scores.ndim != weights.ndim:
        if scores.ndim != weights.ndim + 1:
            raise Exception(
                "scores and weights do not match in terms of dimensionality."
            )
        # multiclass means the scores have the class scores in the last dimension

        new_dims = None
        new_tensor = []
        new_intercept = []
        for class_idx in range(scores.shape[-1]):
            tensor, impurities, intercept = _purify_downstream(
                scores[..., class_idx], weights, tolerance, is_randomized
            )
            new_tensor.append(tensor)
            new_intercept.append(intercept)
            if new_dims is None:
                new_dims = [dims for dims, _ in impurities]
                new_impurities = [[] for _ in impurities]
            for i in range(len(impurities)):
                new_impurities[i].append(impurities[i][1])

        impurities = [
            (key, np.stack(vals, axis=-1, dtype=float))
            for key, vals in zip(new_dims, new_impurities)
        ]
        new_tensor = np.stack(new_tensor, axis=-1, dtype=float)
        new_intercept = np.array(new_intercept, float)
        return new_tensor, impurities, new_intercept
    tensor, impurities, intercept = _purify_downstream(
        scores, weights, tolerance, is_randomized
    )
    return tensor, impurities, np.array([intercept], float)
